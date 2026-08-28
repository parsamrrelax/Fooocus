import os
import sys

# Override Colab's default interactive matplotlib backend which causes errors in standalone python / virtualenv
if os.environ.get('MPLBACKEND') == 'module://matplotlib_inline.backend_inline':
    os.environ['MPLBACKEND'] = 'Agg'

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

from modules.colab_prepare import prepare_colab_environment

prepare_colab_environment()

try:
    import pygit2
    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

    repo = pygit2.Repository(os.path.abspath(os.path.dirname(__file__)))

    branch_name = repo.head.shorthand

    remote_name = 'origin'
    remote = repo.remotes[remote_name]

    remote.fetch()

    local_branch_ref = f'refs/heads/{branch_name}'
    local_branch = repo.lookup_reference(local_branch_ref)

    remote_reference = f'refs/remotes/{remote_name}/{branch_name}'
    remote_commit = repo.revparse_single(remote_reference)

    merge_result, _ = repo.merge_analysis(remote_commit.id)

    if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
        print("Already up-to-date")
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
        local_branch.set_target(remote_commit.id)
        repo.head.set_target(remote_commit.id)
        repo.checkout_tree(repo.get(remote_commit.id))
        repo.reset(local_branch.target, pygit2.GIT_RESET_HARD)
        print("Update succeeded.")
    else:
        print("Cannot fast-forward, merging is required.")

except Exception as e:
    pass

from modules.launch_util import is_installed

if not is_installed("torch") or not is_installed("torchvision"):
    print("Torch is not installed. Please run the setup script.")
    sys.exit(1)

from launch import *
