"""
Validation utilities for input data.
"""

import re
from typing import Optional
from fastapi import HTTPException, status


def validate_file_id(file_id: str) -> str:
    """
    Validate file ID format.
    
    Args:
        file_id: File ID to validate
        
    Returns:
        Validated file ID
        
    Raises:
        HTTPException: If file ID is invalid
    """
    # Check if it's a valid UUID format
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    
    if not re.match(uuid_pattern, file_id.lower()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file ID format"
        )
    
    return file_id


def validate_analysis_id(analysis_id: str) -> str:
    """
    Validate analysis ID format.
    
    Args:
        analysis_id: Analysis ID to validate
        
    Returns:
        Validated analysis ID
        
    Raises:
        HTTPException: If analysis ID is invalid
    """
    return validate_file_id(analysis_id)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove any directory path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove any dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    
    return filename


def validate_pagination(
    page: int = 1,
    page_size: int = 10,
    max_page_size: int = 100
) -> tuple:
    """
    Validate pagination parameters.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        max_page_size: Maximum allowed page size
        
    Returns:
        Tuple of (validated_page, validated_page_size)
        
    Raises:
        HTTPException: If parameters are invalid
    """
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page number must be >= 1"
        )
    
    if page_size < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page size must be >= 1"
        )
    
    if page_size > max_page_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Page size cannot exceed {max_page_size}"
        )
    
    return page, page_size


def validate_medical_value(value: str) -> Optional[float]:
    """
    Validate and convert medical value to float.
    
    Args:
        value: String value to validate
        
    Returns:
        Float value or None if invalid
    """
    try:
        # Remove common separators and convert
        clean_value = value.replace(',', '').strip()
        return float(clean_value)
    except (ValueError, AttributeError):
        return None
