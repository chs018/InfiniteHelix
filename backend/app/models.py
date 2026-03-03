"""
SQLAlchemy database models for Infinite Helix.
Defines the database schema for medical reports and analysis.
"""

from sqlalchemy import Column, String, Integer, DateTime, Text, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

from app.database import Base


def generate_uuid():
    """Generate a unique identifier."""
    return str(uuid.uuid4())


class UploadedFile(Base):
    """Model for uploaded medical report files."""
    
    __tablename__ = "uploaded_files"
    
    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, image, text
    file_size = Column(Integer, nullable=False)  # in bytes
    file_path = Column(String(512), nullable=False)
    upload_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    user_id = Column(String(36), nullable=True)  # For future user authentication
    status = Column(String(50), default="uploaded")  # uploaded, processing, completed, failed
    
    # Relationships
    analyses = relationship("Analysis", back_populates="file", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<UploadedFile(id={self.id}, filename={self.filename})>"


class Analysis(Base):
    """Model for medical report analysis results."""
    
    __tablename__ = "analyses"
    
    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    file_id = Column(String(36), ForeignKey("uploaded_files.id", ondelete="CASCADE"), nullable=False)
    analysis_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # OCR Results
    extracted_text = Column(Text, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    
    # NLP Processing
    entities = Column(JSON, nullable=True)  # Medical entities extracted
    keywords = Column(JSON, nullable=True)  # Important medical terms
    
    # Analysis Status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)  # in seconds
    
    # Relationships
    file = relationship("UploadedFile", back_populates="analyses")
    metrics = relationship("MedicalMetric", back_populates="analysis", cascade="all, delete-orphan")
    insights = relationship("HealthInsight", back_populates="analysis", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Analysis(id={self.id}, file_id={self.file_id}, status={self.status})>"


class MedicalMetric(Base):
    """Model for individual medical test metrics."""
    
    __tablename__ = "medical_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    
    # Metric Information
    metric_name = Column(String(255), nullable=False)  # e.g., "Blood Glucose", "Cholesterol"
    metric_value = Column(String(100), nullable=False)  # The actual value
    metric_unit = Column(String(50), nullable=True)  # e.g., "mg/dL", "mmol/L"
    
    # Reference Ranges
    reference_min = Column(Float, nullable=True)
    reference_max = Column(Float, nullable=True)
    reference_range = Column(String(100), nullable=True)  # Text representation
    
    # Status Assessment
    status = Column(String(50), nullable=True)  # normal, low, high, critical
    severity = Column(String(50), nullable=True)  # mild, moderate, severe
    
    # Additional Context
    category = Column(String(100), nullable=True)  # e.g., "Blood Test", "Urinalysis"
    notes = Column(Text, nullable=True)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="metrics")
    
    def __repr__(self):
        return f"<MedicalMetric(name={self.metric_name}, value={self.metric_value})>"


class HealthInsight(Base):
    """Model for AI-generated health insights and recommendations."""
    
    __tablename__ = "health_insights"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    
    # Insight Information
    insight_type = Column(String(100), nullable=False)  # summary, warning, recommendation
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String(50), nullable=True)  # info, warning, critical
    
    # Prioritization
    priority = Column(Integer, default=0)  # Higher number = higher priority
    is_actionable = Column(Boolean, default=False)
    
    # Metadata
    created_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="insights")
    
    def __repr__(self):
        return f"<HealthInsight(type={self.insight_type}, title={self.title})>"


class AnalysisHistory(Base):
    """Model for tracking user's analysis history and trends."""
    
    __tablename__ = "analysis_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), nullable=True, index=True)  # For future user authentication
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    
    # Summary Information
    report_type = Column(String(100), nullable=True)  # blood test, x-ray, etc.
    report_date = Column(DateTime(timezone=True), nullable=True)  # Date from the report itself
    created_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Quick Access Metadata
    key_findings = Column(JSON, nullable=True)  # Top 3-5 findings for quick display
    overall_status = Column(String(50), nullable=True)  # good, attention_needed, critical
    
    def __repr__(self):
        return f"<AnalysisHistory(id={self.id}, analysis_id={self.analysis_id})>"
