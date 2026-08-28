import importlib.metadata
import os
import re
import sys
from pathlib import Path

import packaging.version
from packaging.requirements import Requirement

from modules.launch_util import run_pip

# Colab pre-installs packages (e.g. numpy 2.x, transformers 5.x) that break Fooocus. Pin them here so
# `python entry_with_update.py` fixes versions before any heavy imports — no runtime restart.
COLAB_REQUIREMENTS = [
    'numpy==1.26.4',
    'starlette>=0.27.0,<1.0.0',
    'transformers>=4.42.4,<4.45.0',
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
        ws_ping_interval=30.0,
        ws_ping_timeout=300.0,
        timeout_keep_alive=300,
        timeout_notify=300,
        ws_per_message_deflate=False
    )'''


def is_colab():
    return 'COLAB_RELEASE_TAG' in os.environ


def is_in_venv():
    return sys.prefix != getattr(sys, 'base_prefix', sys.prefix)


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

    # Fix Colab's MPLBACKEND environment variable for non-interactive backend
    if os.environ.get('MPLBACKEND') == 'module://matplotlib_inline.backend_inline':
        os.environ['MPLBACKEND'] = 'Agg'

    # Colab runs on GCP; Hugging Face Xet CDN (us.gcp.cdn.hf.co) often returns 403.
    # Disable Xet so downloads use the regular Hub/LFS path instead.
    os.environ['HF_HUB_DISABLE_XET'] = '1'

    # If running inside an isolated virtual environment (e.g. uv venv), dependencies are already managed
    if is_in_venv():
        return

    if colab_requirements_met():
        print('[Colab] Dependency versions OK.')
        return

    print('[Colab] Installing pinned dependencies (no runtime restart needed when launched via python)...')
    packages = ' '.join(f'\"{req}\"' for req in COLAB_REQUIREMENTS)
    run_pip(f'install {packages}', desc='Colab dependencies', live=True)


def _gradio_file_path(filename: str):
    """Locate any file inside gradio/ package without importing gradio (avoids Colab kernel breakage)."""
    ver = f'{sys.version_info.major}.{sys.version_info.minor}'
    for base in (
        f'/usr/local/lib/python{ver}/dist-packages',
        f'/usr/local/lib/python{ver}/site-packages',
    ):
        path = Path(base) / 'gradio' / filename
        if path.is_file():
            return path
    try:
        root = Path(importlib.metadata.distribution('gradio').locate_file(''))
        path = root / 'gradio' / filename
        if path.is_file():
            return path
    except Exception:
        pass
    return None


def patch_gradio_websocket_limits():
    """Raise Gradio/uvicorn WebSocket limits and patch queue resilience so UI does not freeze or disconnect."""
    if not is_colab():
        return

    net_path = _gradio_file_path('networking.py')
    if net_path is not None:
        text = net_path.read_text()
        if _GRADIO_WS_MARKER in text:
            print(f'[Colab] Gradio WebSocket limits already patched: {net_path}')
        else:
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
            if n == 1:
                net_path.write_text(new_text)
                print(f'[Colab] Patched Gradio WebSocket limits in {net_path}')
            else:
                # Fallback: replace uvicorn.Config call directly
                start_idx = text.find('config = uvicorn.Config(')
                if start_idx != -1:
                    end_idx = text.find(')', start_idx)
                    if end_idx != -1:
                        patched = text[:start_idx] + _GRADIO_WS_CONFIG + text[end_idx + 1:]
                        net_path.write_text(patched)
                        print(f'[Colab] Patched Gradio WebSocket limits (fallback) in {net_path}')
                else:
                    print(f'[Colab] Could not patch uvicorn.Config in {net_path}; leaving file unchanged.')

    # Fix Gradio 3.x Queue crash: 'AsyncRequest' object has no attribute '_json_response_data'
    utils_path = _gradio_file_path('utils.py')
    if utils_path is not None:
        text = utils_path.read_text()
        if 'return self._json_response_data' in text:
            new_text = text.replace(
                'return self._json_response_data',
                "return getattr(self, '_json_response_data', {})"
            )
            utils_path.write_text(new_text)
            print(f'[Colab] Patched Gradio AsyncRequest resilience in {utils_path}')
