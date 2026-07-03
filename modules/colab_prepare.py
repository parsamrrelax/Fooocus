import importlib.metadata
import os

import packaging.version
from packaging.requirements import Requirement

from modules.launch_util import run_pip

# Colab pre-installs packages (e.g. numpy 2.x) that break Fooocus. Pin them here so
# `python entry_with_update.py` fixes versions before any heavy imports — no runtime restart.
COLAB_REQUIREMENTS = [
    'numpy<2.0.0',
    'cupy-cuda12x<14.0',
    'pygit2==1.15.1',
    'starlette>=0.27.0,<1.0.0',
]


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

    if colab_requirements_met():
        print('[Colab] Dependency versions OK.')
        return

    print('[Colab] Installing pinned dependencies (no runtime restart needed when launched via python)...')
    packages = ' '.join(f'"{req}"' for req in COLAB_REQUIREMENTS)
    run_pip(f'install {packages}', desc='Colab dependencies', live=True)
