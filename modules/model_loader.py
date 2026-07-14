import os
import re
import shutil
import subprocess
from urllib.parse import urlparse, unquote
from typing import Optional, Tuple


_HF_RESOLVE_RE = re.compile(
    r'^https?://(?:huggingface\.co|hf\.co)/'
    r'(?P<repo>[^/]+/[^/]+)/resolve/'
    r'(?P<revision>[^/]+)/'
    r'(?P<filename>.+)$'
)


def _is_huggingface_host(url: str) -> bool:
    host = (urlparse(url).hostname or '').lower()
    return host in ('huggingface.co', 'hf.co') or host.endswith('.huggingface.co')


def _headers_for_url(url: str, hf_token: str) -> dict:
    if hf_token and _is_huggingface_host(url):
        return {'Authorization': f'Bearer {hf_token}'}
    return {}


def _parse_hf_resolve_url(url: str) -> Optional[Tuple[str, str, str]]:
    """Return (repo_id, revision, filename) for Hub resolve URLs, else None."""
    # Normalize custom HF mirrors back to a parseable huggingface.co URL.
    domain = os.environ.get('HF_MIRROR', 'https://huggingface.co').rstrip('/')
    normalized = url
    if domain != 'https://huggingface.co' and url.startswith(domain + '/'):
        normalized = 'https://huggingface.co/' + url[len(domain) + 1:]
    match = _HF_RESOLVE_RE.match(normalized.split('?', 1)[0])
    if not match:
        return None
    return match.group('repo'), unquote(match.group('revision')), unquote(match.group('filename'))


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    If the HF_TOKEN environment variable is set (e.g. via --hf-token or the Colab
    notebook's token field) and the file is being fetched from Hugging Face, the
    token is sent with the request. This lets Hugging Face grant a higher rate
    limit / faster download quota instead of the default anonymous one.

    Returns the path to the downloaded file.
    """
    is_hf_url = url.startswith('https://huggingface.co') or url.startswith('https://hf.co')
    domain = os.environ.get('HF_MIRROR', 'https://huggingface.co').rstrip('/')
    url = url.replace('https://huggingface.co', domain, 1).replace('https://hf.co', domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    hf_token = os.environ.get('HF_TOKEN', '').strip() if is_hf_url else ''

    if os.path.exists(cached_file):
        return cached_file

    print(f'Downloading: "{url}" to {cached_file}\n')

    # Colab/GCP often gets us.gcp.cdn.hf.co Xet redirects that 403. Prefer the Hub
    # client with Xet disabled for Hugging Face resolve URLs.
    hf_parts = _parse_hf_resolve_url(url)
    if hf_parts is not None:
        try:
            _download_hf_via_hub(hf_parts, cached_file, hf_token=hf_token)
            return cached_file
        except Exception as e:
            print(f'huggingface_hub download failed ({e}); falling back to direct download.')

    download_url = _resolve_download_url(url, hf_token=hf_token)

    if shutil.which('aria2c'):
        try:
            connections = '16' if hf_token else '4'
            cmd = [
                'aria2c', '--quiet=true', '-c',
                '-x', connections, '-s', connections, '-k', '8M',
                '--retry-wait=5', '--max-tries=10',
                '-d', model_dir, '-o', file_name, download_url,
            ]
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            return cached_file
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or '').strip()
            if detail:
                print(f'aria2c failed ({detail[:300]}), falling back to default downloader.')
            else:
                print('aria2c failed, falling back to default downloader.')

    _download_url_to_file(download_url, cached_file, hf_token=hf_token, progress=progress)
    return cached_file


def _download_hf_via_hub(hf_parts: Tuple[str, str, str], cached_file: str, *, hf_token: str) -> None:
    """Download via huggingface_hub, avoiding broken GCP Xet CDN redirects."""
    # Must be set before importing huggingface_hub internals that read it once.
    os.environ['HF_HUB_DISABLE_XET'] = '1'

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError('huggingface_hub is not installed') from e

    repo_id, revision, filename = hf_parts
    token = hf_token or None
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )
    if os.path.abspath(path) != os.path.abspath(cached_file):
        os.makedirs(os.path.dirname(cached_file), exist_ok=True)
        shutil.copy2(path, cached_file)


def _resolve_download_url(url: str, *, hf_token: str, max_redirects: int = 10) -> str:
    """Follow redirects; send Authorization only to huggingface.co hosts."""
    import requests

    current = url
    for _ in range(max_redirects):
        response = requests.get(
            current,
            headers=_headers_for_url(current, hf_token),
            stream=True,
            timeout=60,
            allow_redirects=False,
        )
        if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get('Location')
            response.close()
            if not location:
                response.raise_for_status()
            current = requests.compat.urljoin(current, location)
            continue
        response.raise_for_status()
        response.close()
        return current
    raise RuntimeError(f'Too many redirects while resolving download URL: {url}')


def _download_url_to_file(url: str, dst: str, *, hf_token: str = '', progress: bool = True) -> None:
    """Download `url` to `dst`. Auth is only sent to huggingface.co hosts."""
    import uuid
    import requests
    from tqdm import tqdm

    dst = os.path.abspath(dst)
    dst_dir = os.path.dirname(dst)
    tmp_dst = os.path.join(dst_dir, f'{os.path.basename(dst)}.{uuid.uuid4().hex}.partial')

    try:
        with requests.get(
                url,
                headers=_headers_for_url(url, hf_token),
                stream=True,
                timeout=60,
                allow_redirects=False,
        ) as response:
            if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                raise RuntimeError(
                    f'Unexpected redirect while downloading: {url} -> {response.headers.get("Location")}'
                )
            response.raise_for_status()
            total_size = int(response.headers.get('Content-Length', 0))
            with open(tmp_dst, 'wb') as f, tqdm(
                    total=total_size if total_size > 0 else None,
                    disable=not progress,
                    unit='B', unit_scale=True, unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        shutil.move(tmp_dst, dst)
    finally:
        if os.path.exists(tmp_dst):
            os.remove(tmp_dst)
