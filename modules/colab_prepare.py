import importlib.metadata
import os
import re
import sys
from pathlib import Path

import packaging.version
from packaging.requirements import Requirement

from modules.launch_util import run_pip

# Colab pre-installs packages (e.g. numpy 2.x) that break Fooocus. Pin them here so
# `python entry_with_update.py` fixes versions before any heavy imports — no runtime restart.
COLAB_REQUIREMENTS = [
    'numpy<2.0.0',
    'cupy-cuda12x<14.0',
    'starlette>=0.27.0,<1.0.0',
    'huggingface_hub',
]

# Gradio default WebSocket limits drop large image edits on Colab (browser error 1006).
_GRADIO_WS_MARKER = 'ws_max_size=1 * 1024 * 1024 * 1024'
_GRADIO_WS_CONFIG = '''config = uvicorn.Config(
        app=app,
        port=port,
        host=host,
        log_level="warning",
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ws_max_size=1 * 1024 * 1024 * 1024,  # Setting max websocket size to be 1 GB
        ws_max_queue=64,
        ws_ping_interval=60.0,
        ws_ping_timeout=10.0,
        ws_per_message_deflate=False,
        reload=True,
        timeout_notify=120
    )'''


def is_colab():
    return 'COLAB_RELEASE_TAG' in os.environ


def colab_requirements_met():
    for req_str in COLAB_REQUIREMENTS:
        requirement = Requirement(req_str)
        package = requirement.name
        try:
            version_installed = importlib.metadata.version(package)
            installed_version = packaging.version.parse(version_installed)
            if installed_version not in requirement.specifier:
                print(f'[Colab] Version mismatch for {package}: '
                      f'installed {version_installed}, need {requirement.specifier}')
                return False
        except Exception as e:
            print(f'[Colab] Missing or unreadable package {package}: {e}')
            return False
    return True


def prepare_colab_environment():
    if not is_colab():
        return

    # Colab runs on GCP; Hugging Face Xet CDN (us.gcp.cdn.hf.co) often returns 403.
    # Disable Xet so downloads use the regular Hub/LFS path instead.
    os.environ['HF_HUB_DISABLE_XET'] = '1'

    if colab_requirements_met():
        print('[Colab] Dependency versions OK.')
        return

    print('[Colab] Installing pinned dependencies (no runtime restart needed when launched via python)...')
    packages = ' '.join(f'"{req}"' for req in COLAB_REQUIREMENTS)
    run_pip(f'install {packages}', desc='Colab dependencies', live=True)


def _gradio_networking_path():
    """Locate gradio/networking.py without importing gradio (avoids Colab kernel breakage)."""
    ver = f'{sys.version_info.major}.{sys.version_info.minor}'
    for base in (
        f'/usr/local/lib/python{ver}/dist-packages',
        f'/usr/local/lib/python{ver}/site-packages',
    ):
        path = Path(base) / 'gradio' / 'networking.py'
        if path.is_file():
            return path
    try:
        root = Path(importlib.metadata.distribution('gradio').locate_file(''))
        path = root / 'gradio' / 'networking.py'
        if path.is_file():
            return path
    except Exception:
        pass
    return None


def patch_gradio_websocket_limits():
    """Raise Gradio/uvicorn WebSocket limits so large image edits do not hit error 1006."""
    if not is_colab():
        return

    path = _gradio_networking_path()
    if path is None:
        print('[Colab] Gradio networking.py not found; skipping WebSocket patch.')
        return

    text = path.read_text()
    if _GRADIO_WS_MARKER in text:
        print(f'[Colab] Gradio WebSocket limits already patched: {path}')
        return

    pattern = re.compile(
        r'config\s*=\s*uvicorn\.Config\(\s*'
        r'app=app,\s*'
        r'port=port,\s*'
        r'host=host,\s*'
        r'log_level="warning",\s*'
        r'ssl_keyfile=ssl_keyfile,\s*'
        r'ssl_certfile=ssl_certfile,\s*'
        r'ssl_keyfile_password=ssl_keyfile_password,\s*'
        r'.*?'
        r'\)',
        re.DOTALL,
    )
    new_text, n = pattern.subn(_GRADIO_WS_CONFIG, text, count=1)
    if n != 1:
        print(f'[Colab] Could not patch uvicorn.Config in {path}; leaving file unchanged.')
        return

    path.write_text(new_text)
    print(f'[Colab] Patched Gradio WebSocket limits in {path}')
