"""
File manager module — Handle upload storage and metadata.
"""
import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "../uploads")
UPLOAD_METADATA_FILE = os.path.join(UPLOADS_DIR, ".metadata.json")


def ensure_uploads_dir():
    """Create uploads directory if it doesn't exist."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)


def load_upload_metadata() -> dict:
    """Load metadata about uploaded files."""
    ensure_uploads_dir()
    if os.path.exists(UPLOAD_METADATA_FILE):
        try:
            with open(UPLOAD_METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading upload metadata: {e}")
    return {}


def save_upload_metadata(metadata: dict):
    """Save metadata about uploaded files."""
    ensure_uploads_dir()
    try:
        with open(UPLOAD_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving upload metadata: {e}")


def file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def register_upload(original_filename: str, file_path: str, num_chunks: int) -> dict:
    """
    Register an uploaded file in metadata.
    Returns the metadata entry.
    """
    ensure_uploads_dir()
    metadata = load_upload_metadata()
    
    file_hash_val = file_hash(file_path)
    entry = {
        "filename": original_filename,
        "path": file_path,
        "file_hash": file_hash_val,
        "upload_time": datetime.utcnow().isoformat(),
        "num_chunks": num_chunks,
        "file_size": os.path.getsize(file_path),
    }
    
    # Use file hash as key to deduplicate
    metadata[file_hash_val] = entry
    save_upload_metadata(metadata)
    
    logger.info(f"Registered upload: {original_filename} (hash: {file_hash_val})")
    return entry


def get_uploaded_files() -> list:
    """Get list of all uploaded files."""
    metadata = load_upload_metadata()
    return list(metadata.values())


def cleanup_upload(file_hash_val: str):
    """Remove an uploaded file and its metadata."""
    ensure_uploads_dir()
    metadata = load_upload_metadata()
    
    if file_hash_val in metadata:
        entry = metadata[file_hash_val]
        file_path = entry.get("path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
        
        del metadata[file_hash_val]
        save_upload_metadata(metadata)
        logger.info(f"Removed metadata for: {file_hash_val}")


def save_uploaded_file(file_obj, original_filename: str) -> str:
    """
    Save an uploaded file to the uploads directory.
    Returns the path to the saved file.
    """
    ensure_uploads_dir()
    
    # Sanitize filename
    from werkzeug.utils import secure_filename
    safe_name = secure_filename(original_filename)
    
    # If sanitization removes too much, use a fallback
    if not safe_name:
        safe_name = f"upload_{datetime.utcnow().timestamp()}"
    
    file_path = os.path.join(UPLOADS_DIR, safe_name)
    file_obj.save(file_path)
    
    logger.info(f"Saved uploaded file: {file_path}")
    return file_path
