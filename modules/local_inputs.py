import os
import re
import shutil
import subprocess
import zipfile
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
ARCHIVE_EXTENSIONS = ('.zip', '.7z', '.rar', '.tar.gz', '.tar')

_current_selected_image_path = None


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def get_default_input_dir() -> str:
    # If in Colab environment or directory exists
    colab_dir = '/content/Fooocus/inputs'
    if os.path.exists('/content') or os.path.exists(colab_dir):
        os.makedirs(colab_dir, exist_ok=True)
        return colab_dir
    local_dir = os.path.abspath('./inputs')
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def resolve_input_dir(folder: str | None = None) -> str:
    if not folder or not folder.strip():
        return get_default_input_dir()
    folder = os.path.expanduser(folder.strip())
    os.makedirs(folder, exist_ok=True)
    return os.path.abspath(folder)


def get_zip_files(folder: str | None = None) -> list[str]:
    target_dir = resolve_input_dir(folder)
    if not os.path.isdir(target_dir):
        return []
    archives = []
    try:
        for f in os.listdir(target_dir):
            if f.lower().endswith(ARCHIVE_EXTENSIONS):
                archives.append(f)
    except Exception as e:
        print(f"[LocalInputs] Error listing zip files: {e}")
        return []
    # Sort by modification time (most recent first)
    archives.sort(key=lambda x: os.path.getmtime(os.path.join(target_dir, x)), reverse=True)
    return archives


def get_latest_zip_file(folder: str | None = None) -> str | None:
    files = get_zip_files(folder)
    return files[0] if len(files) > 0 else None


def extract_zip_archive(zip_filename: str | None, password: str | None = None, folder: str | None = None) -> tuple[bool, str, int]:
    """
    Extracts a password-protected (or plain) archive into the target folder.
    Supports AES-256 encrypted zips via 7z / unzip CLI and fallback to zipfile.
    Returns (success, message, extracted_image_count).
    """
    if not zip_filename:
        return False, "No zip file selected.", 0

    base_dir = resolve_input_dir(folder)
    zip_path = os.path.join(base_dir, zip_filename) if not os.path.isabs(zip_filename) else zip_filename

    if not os.path.isfile(zip_path):
        return False, f"Archive file not found: {zip_path}", 0

    password = password.strip() if password else None
    
    # Destination directory: subfolder 'extracted' or directly in input folder
    stem = os.path.splitext(os.path.basename(zip_path))[0]
    dest_dir = os.path.join(base_dir, stem)
    os.makedirs(dest_dir, exist_ok=True)

    extracted_ok = False
    err_detail = ""

    # Attempt 1: 7z (Supports AES-256 WinZip, ZipCrypto, 7z, etc.)
    if shutil.which("7z"):
        cmd = ["7z", "x"]
        if password:
            cmd.append(f"-p{password}")
        else:
            cmd.append("-p")  # empty/prompt bypass
        cmd.extend([f"-o{dest_dir}", "-y", zip_path])
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                extracted_ok = True
            else:
                err_detail = res.stderr or res.stdout
                print(f"[LocalInputs] 7z failed (code {res.returncode}): {err_detail}")
        except Exception as e:
            err_detail = str(e)
            print(f"[LocalInputs] 7z execution exception: {e}")

    # Attempt 2: unzip CLI
    if not extracted_ok and shutil.which("unzip"):
        cmd = ["unzip"]
        if password:
            cmd.extend(["-P", password])
        cmd.extend(["-o", zip_path, "-d", dest_dir])
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                extracted_ok = True
            else:
                err_detail = res.stderr or res.stdout
                print(f"[LocalInputs] unzip failed (code {res.returncode}): {err_detail}")
        except Exception as e:
            err_detail = str(e)
            print(f"[LocalInputs] unzip execution exception: {e}")

    # Attempt 3: Standard Python zipfile module fallback
    if not extracted_ok and zip_path.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                pwd_bytes = password.encode('utf-8') if password else None
                zf.extractall(dest_dir, pwd=pwd_bytes)
                extracted_ok = True
        except RuntimeError as e:
            err_detail = f"Bad password or encryption error: {e}"
        except Exception as e:
            err_detail = str(e)

    if not extracted_ok:
        msg = "❌ Extraction failed! Please verify the password."
        if err_detail:
            if "wrong password" in err_detail.lower() or "bad password" in err_detail.lower() or "incorrect password" in err_detail.lower():
                msg = "❌ Incorrect password! Please check the password and try again."
        return False, msg, 0

    # Count extracted images
    images = get_image_files(base_dir)
    return True, f"✅ Successfully extracted `{zip_filename}` into `{os.path.basename(dest_dir)}/` ({len(images)} total images found).", len(images)


def get_image_files(folder: str | None = None) -> list[str]:
    """
    Recursively scans the input directory for all supported image files.
    Returns relative paths sorted naturally.
    """
    target_dir = resolve_input_dir(folder)
    if not os.path.isdir(target_dir):
        return []

    images = []
    try:
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, target_dir)
                    images.append(rel_path)
    except Exception as e:
        print(f"[LocalInputs] Error scanning images: {e}")
        return []

    images.sort(key=natural_sort_key)
    return images


def load_image_for_canvas(relative_path: str | None, folder: str | None = None, max_preview_edge: int = 1920) -> np.ndarray | None:
    """
    Loads an image from the local inputs directory for the inpaint canvas.
    Downscales if longer edge exceeds max_preview_edge to keep canvas interactions
    and transfer fast over slow connections.
    """
    global _current_selected_image_path
    if not relative_path:
        return None

    base_dir = resolve_input_dir(folder)
    full_path = os.path.join(base_dir, relative_path) if not os.path.isabs(relative_path) else relative_path

    if not os.path.isfile(full_path):
        print(f"[LocalInputs] Image not found: {full_path}")
        return None

    try:
        pil_img = Image.open(full_path)
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size

        # Cap preview size if huge to keep Gradio network serialization light
        if max_preview_edge and max_preview_edge > 0:
            max_dim = max(w, h)
            if max_dim > max_preview_edge:
                scale = max_preview_edge / float(max_dim)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                lanczos = getattr(Image, 'Resampling', Image).LANCZOS
                pil_img = pil_img.resize((new_w, new_h), resample=lanczos)

        _current_selected_image_path = os.path.abspath(full_path)
        return np.array(pil_img)
    except Exception as e:
        print(f"[LocalInputs] Failed to load image {full_path}: {e}")
        return None


def get_current_selected_image_path() -> str | None:
    global _current_selected_image_path
    return _current_selected_image_path


def clear_selected_image_path():
    global _current_selected_image_path
    _current_selected_image_path = None


def load_original_disk_image(path: str | None) -> np.ndarray | None:
    """
    Loads the pristine full-resolution image directly from disk.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img)
    except Exception as e:
        print(f"[LocalInputs] Error loading original image from {path}: {e}")
        return None
