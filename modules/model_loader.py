import os
import shutil
import subprocess
from urllib.parse import urlparse
from typing import Optional


def _is_huggingface_host(url: str) -> bool:
    host = (urlparse(url).hostname or '').lower()
    return host == 'huggingface.co' or host.endswith('.huggingface.co')


def _headers_for_url(url: str, hf_token: str) -> dict:
    if hf_token and _is_huggingface_host(url):
        return {'Authorization': f'Bearer {hf_token}'}
    return {}


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
    is_hf_url = url.startswith("https://huggingface.co")
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    hf_token = os.environ.get("HF_TOKEN", "").strip() if is_hf_url else ""

    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')

        # Resolve redirects with auth only on huggingface.co. Signed CDN URLs
        # (cdn.hf.co / xet-bridge) return 403 if Authorization is forwarded.
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
                # Never attach HF auth to aria2: it would be sent to the CDN too.
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                )
                return cached_file
            except subprocess.CalledProcessError as e:
                detail = (e.stderr or '').strip()
                if detail:
                    print(f"aria2c failed ({detail[:300]}), falling back to default downloader.")
                else:
                    print("aria2c failed, falling back to default downloader.")

        _download_url_to_file(download_url, cached_file, hf_token=hf_token, progress=progress)
    return cached_file


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
            # Final hop should already be resolved; still reject surprise redirects.
            if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                raise RuntimeError(f'Unexpected redirect while downloading: {url} -> {response.headers.get("Location")}')
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
