"""API input validation for uploaded files."""

from werkzeug.datastructures import FileStorage

ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}

MAX_IMAGE_BYTES = 10 * 1024 * 1024    # 10 MB
MAX_VIDEO_BYTES = 200 * 1024 * 1024   # 200 MB


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def validate_upload(file: FileStorage) -> tuple[str, str]:
    """Validate an uploaded file. Returns (media_type, extension).

    Raises ValueError with a user-facing message on failure.
    """
    if not file or not file.filename:
        raise ValueError("No file selected.")

    ext = _ext(file.filename)

    if ext in ALLOWED_IMAGE_EXT:
        media_type = "image"
        max_bytes = MAX_IMAGE_BYTES
    elif ext in ALLOWED_VIDEO_EXT:
        media_type = "video"
        max_bytes = MAX_VIDEO_BYTES
    else:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT))
        raise ValueError(f"Unsupported file type '.{ext}'. Allowed: {allowed}.")

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)

    if size > max_bytes:
        limit_mb = max_bytes // (1024 * 1024)
        actual_mb = size // (1024 * 1024)
        raise ValueError(f"File too large ({actual_mb} MB). Limit is {limit_mb} MB.")

    return media_type, ext
