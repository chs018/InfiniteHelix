"""
Upload router for file upload endpoints.
Handles medical report file uploads with validation.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from sqlalchemy.orm import Session
import logging
from datetime import datetime

from app.database import get_db
from app.schemas import FileUploadResponse, ErrorResponse
from app.models import UploadedFile
from app.utils.file_handler import file_handler
from app.utils.validators import sanitize_filename
from app.routers.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["upload"])


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload medical report file",
    description="Upload a medical report file (PDF, image, or text) for analysis"
)
async def upload_file(
    file: UploadFile = File(..., description="Medical report file to upload"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Upload a medical report file.
    
    Accepts PDF, PNG, JPG, JPEG, and TXT files up to 10MB.
    Returns file information including unique file ID for analysis.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Sanitize filename
        original_filename = sanitize_filename(file.filename)
        
        # Check file extension
        if not file_handler.is_allowed_extension(original_filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not supported. Allowed types: PDF, PNG, JPG, JPEG, TXT"
            )
        
        # Save file
        file_path, unique_filename, file_size = await file_handler.save_upload_file(file)
        
        # Get file type
        file_extension = file_handler.get_file_extension(original_filename)
        
        # Create database record
        db_file = UploadedFile(
            filename=unique_filename,
            original_filename=original_filename,
            file_type=file_extension,
            file_size=file_size,
            file_path=file_path,
            status="uploaded"
        )
        
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        
        logger.info(f"File uploaded successfully: {db_file.id}")
        
        # Return response
        return FileUploadResponse(
            file_id=db_file.id,
            filename=db_file.original_filename,
            file_type=db_file.file_type,
            file_size=db_file.file_size,
            upload_date=db_file.upload_date,
            status=db_file.status,
            message="File uploaded successfully. Use the file_id to analyze the report."
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while uploading the file"
        )


@router.delete(
    "/upload/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete uploaded file",
    description="Delete an uploaded file and its analysis results"
)
async def delete_upload(
    file_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete an uploaded file and all associated data.
    """
    try:
        # Get file from database
        db_file = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
        
        if not db_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Delete physical file
        await file_handler.delete_file(db_file.file_path)
        
        # Delete database record (cascades to analyses)
        db.delete(db_file)
        db.commit()
        
        logger.info(f"File deleted: {file_id}")
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the file"
        )
