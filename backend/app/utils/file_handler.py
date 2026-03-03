"""
File handling utilities for secure upload and processing.
"""

import os
import uuid
import shutil
import aiofiles
from pathlib import Path
from typing import Tuple, Optional
import logging
from fastapi import UploadFile

from app.config import settings

logger = logging.getLogger(__name__)


class FileHandler:
    """Handler for file upload and storage operations."""
    
    def __init__(self):
        """Initialize file handler with upload directory."""
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """
        Generate a unique filename while preserving the extension.
        
        Args:
            original_filename: Original uploaded filename
            
        Returns:
            Unique filename
        """
        file_extension = Path(original_filename).suffix
        unique_id = str(uuid.uuid4())
        return f"{unique_id}{file_extension}"
    
    def get_file_extension(self, filename: str) -> str:
        """
        Get file extension without the dot.
        
        Args:
            filename: Filename to extract extension from
            
        Returns:
            File extension (lowercase, without dot)
        """
        return Path(filename).suffix.lstrip('.').lower()
    
    def is_allowed_extension(self, filename: str) -> bool:
        """
        Check if file extension is allowed.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if extension is allowed, False otherwise
        """
        extension = self.get_file_extension(filename)
        return extension in settings.allowed_extensions
    
    async def save_upload_file(
        self,
        upload_file: UploadFile,
        max_size: Optional[int] = None
    ) -> Tuple[str, str, int]:
        """
        Save an uploaded file to disk.
        
        Args:
            upload_file: FastAPI UploadFile object
            max_size: Maximum file size in bytes (None for settings default)
            
        Returns:
            Tuple of (file_path, unique_filename, file_size)
        """
        try:
            # Validate extension
            if not self.is_allowed_extension(upload_file.filename):
                raise ValueError(
                    f"File type not allowed. Allowed types: {', '.join(settings.allowed_extensions)}"
                )
            
            # Generate unique filename
            unique_filename = self.generate_unique_filename(upload_file.filename)
            file_path = self.upload_dir / unique_filename
            
            # Save file
            file_size = 0
            max_allowed_size = max_size or settings.max_upload_size
            
            async with aiofiles.open(file_path, 'wb') as out_file:
                while True:
                    chunk = await upload_file.read(8192)  # Read in 8KB chunks
                    if not chunk:
                        break
                    
                    file_size += len(chunk)
                    
                    # Check size limit
                    if file_size > max_allowed_size:
                        # Remove partially saved file
                        await self.delete_file(str(file_path))
                        raise ValueError(
                            f"File size exceeds maximum allowed size of "
                            f"{max_allowed_size / (1024 * 1024):.1f}MB"
                        )
                    
                    await out_file.write(chunk)
            
            logger.info(f"File saved: {unique_filename} ({file_size} bytes)")
            
            return str(file_path), unique_filename, file_size
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from disk.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {}
            
            stat = path.stat()
            
            return {
                'filename': path.name,
                'extension': path.suffix.lstrip('.').lower(),
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'exists': True
            }
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return {}
    
    async def cleanup_old_files(self, days: int = 7):
        """
        Clean up files older than specified days.
        
        Args:
            days: Number of days to keep files
        """
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 60 * 60)
            
            deleted_count = 0
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old files")
            
        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}")


# Global file handler instance
file_handler = FileHandler()
