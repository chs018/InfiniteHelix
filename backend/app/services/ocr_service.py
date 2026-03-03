"""
OCR Service for text extraction from medical reports.
Uses Tesseract OCR for image and PDF processing.
"""

import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
import logging
from typing import Tuple, Optional
import tempfile

from app.config import settings

logger = logging.getLogger(__name__)


class OCRService:
    """Service for extracting text from images and PDFs."""
    
    def __init__(self):
        """Initialize OCR service with Tesseract configuration."""
        if settings.tesseract_path and settings.tesseract_path != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path
    
    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            logger.info(f"Extracting text from image: {image_path}")
            
            # Open and preprocess image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Perform OCR with detailed data
            ocr_data = pytesseract.image_to_data(
                image,
                lang=settings.ocr_language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=settings.ocr_language
            )
            
            # Calculate average confidence
            confidences = [
                float(conf) for conf in ocr_data['conf']
                if conf != '-1' and str(conf).replace('.', '').isdigit()
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.info(f"Text extraction completed. Confidence: {avg_confidence:.2f}%")
            
            return text.strip(), avg_confidence
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            raise Exception(f"OCR processing failed: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, float]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            
            all_text = []
            all_confidences = []
            
            # Process each page
            for i, image in enumerate(images, 1):
                logger.info(f"Processing PDF page {i}/{len(images)}")
                
                # Save image temporarily
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image.save(tmp.name, 'PNG')
                    tmp_path = tmp.name
                
                try:
                    # Extract text from page
                    text, confidence = self.extract_text_from_image(tmp_path)
                    all_text.append(text)
                    all_confidences.append(confidence)
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            # Combine results
            combined_text = "\n\n".join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            logger.info(f"PDF text extraction completed. Average confidence: {avg_confidence:.2f}%")
            
            return combined_text.strip(), avg_confidence
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"PDF OCR processing failed: {str(e)}")
    
    def extract_text_from_file(self, file_path: str, file_type: str) -> Tuple[str, float]:
        """
        Extract text from any supported file type.
        
        Args:
            file_path: Path to the file
            file_type: Type of file (pdf, png, jpg, jpeg, txt)
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            file_type = file_type.lower()
            
            if file_type == 'txt':
                # Read text file directly
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return text.strip(), 100.0
            
            elif file_type == 'pdf':
                return self.extract_text_from_pdf(file_path)
            
            elif file_type in ['png', 'jpg', 'jpeg']:
                return self.extract_text_from_image(file_path)
            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error extracting text from file: {str(e)}")
            raise


# Global OCR service instance
ocr_service = OCRService()
