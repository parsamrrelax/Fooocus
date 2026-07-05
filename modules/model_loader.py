import os
import shutil
import subprocess
from urllib.parse import urlparse
from typing import Optional


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

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    auth_header = None
    if is_hf_url and hf_token:
        auth_header = f'Authorization: Bearer {hf_token}'

    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')

        if shutil.which('aria2c'):
            try:
                # A token grants a much higher Hugging Face rate limit, so we can
                # also afford to open more parallel connections to speed things up.
                connections = '16' if auth_header else '4'
                cmd = ['aria2c', '--quiet=true', '-c', '-x', connections, '-s', connections, '-k', '8M',
                       '--retry-wait=5', '--max-tries=0']
                if auth_header:
                    cmd += ['--header', auth_header]
                cmd += ['-d', model_dir, '-o', file_name, url]
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return cached_file
            except subprocess.CalledProcessError:
                print("aria2c failed, falling back to default downloader.")

        if auth_header:
            _download_url_to_file_with_headers(url, cached_file, headers={'Authorization': f'Bearer {hf_token}'}, progress=progress)
        else:
            from torch.hub import download_url_to_file
            download_url_to_file(url, cached_file, progress=progress)
    return cached_file


def _download_url_to_file_with_headers(url: str, dst: str, *, headers: dict, progress: bool = True) -> None:
    """Same behaviour as `torch.hub.download_url_to_file`, but supports custom
    request headers (needed to send the Hugging Face token when aria2c isn't
    available).
    """
    import uuid
    import requests
    from tqdm import tqdm

    dst = os.path.abspath(dst)
    dst_dir = os.path.dirname(dst)
    tmp_dst = os.path.join(dst_dir, f'{os.path.basename(dst)}.{uuid.uuid4().hex}.partial')

    try:
        with requests.get(url, headers=headers, stream=True, timeout=30) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('Content-Length', 0))
            with open(tmp_dst, 'wb') as f, tqdm(
                    total=total_size if total_size > 0 else None,
                    disable=not progress,
                    unit='B', unit_scale=True, unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        shutil.move(tmp_dst, dst)
    finally:
        if os.path.exists(tmp_dst):
            os.remove(tmp_dst)
