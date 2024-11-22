import os
from fastapi import UploadFile


def save_file(file: UploadFile, upload_dir: str) -> str:
    """
    Save an uploaded file to the specified directory.
    """
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return file_path
