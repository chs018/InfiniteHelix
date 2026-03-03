"""
Pydantic schemas for request/response validation.
Defines data transfer objects for the API.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums for status values
class FileStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MetricStatus(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    CRITICAL = "critical"


class InsightType(str, Enum):
    SUMMARY = "summary"
    WARNING = "warning"
    RECOMMENDATION = "recommendation"
    INFO = "info"


# File Upload Schemas
class FileUploadResponse(BaseModel):
    """Response schema for file upload."""
    file_id: str
    filename: str
    file_type: str
    file_size: int
    upload_date: datetime
    status: str
    message: str = "File uploaded successfully"
    
    class Config:
        from_attributes = True


# Medical Metric Schemas
class MedicalMetricBase(BaseModel):
    """Base schema for medical metrics."""
    metric_name: str
    metric_value: str
    metric_unit: Optional[str] = None
    reference_min: Optional[float] = None
    reference_max: Optional[float] = None
    reference_range: Optional[str] = None
    status: Optional[str] = None
    severity: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None


class MedicalMetricCreate(MedicalMetricBase):
    """Schema for creating a medical metric."""
    analysis_id: str


class MedicalMetricResponse(MedicalMetricBase):
    """Response schema for medical metric."""
    id: int
    analysis_id: str
    
    class Config:
        from_attributes = True


# Health Insight Schemas
class HealthInsightBase(BaseModel):
    """Base schema for health insights."""
    insight_type: str
    title: str
    description: str
    severity: Optional[str] = None
    priority: int = 0
    is_actionable: bool = False


class HealthInsightCreate(HealthInsightBase):
    """Schema for creating a health insight."""
    analysis_id: str


class HealthInsightResponse(HealthInsightBase):
    """Response schema for health insight."""
    id: int
    analysis_id: str
    created_date: datetime
    
    class Config:
        from_attributes = True


# Analysis Schemas
class AnalysisBase(BaseModel):
    """Base schema for analysis."""
    file_id: str
    extracted_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    entities: Optional[Dict[str, Any]] = None
    keywords: Optional[List[str]] = None
    status: str = "pending"


class AnalysisCreate(AnalysisBase):
    """Schema for creating an analysis."""
    pass


class AnalysisResponse(AnalysisBase):
    """Response schema for analysis."""
    id: str
    analysis_date: datetime
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    metrics: List[MedicalMetricResponse] = []
    insights: List[HealthInsightResponse] = []
    
    class Config:
        from_attributes = True


class AnalysisSummary(BaseModel):
    """Simplified analysis summary for list views."""
    id: str
    file_id: str
    filename: str
    analysis_date: datetime
    status: str
    metrics_count: int
    insights_count: int
    overall_status: Optional[str] = None
    
    class Config:
        from_attributes = True


# Analysis Request Schema
class AnalysisRequest(BaseModel):
    """Request schema for triggering analysis."""
    file_id: str
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional analysis parameters"
    )


# History Schemas
class HistoryItemResponse(BaseModel):
    """Response schema for history list item."""
    id: int
    analysis_id: str
    report_type: Optional[str] = None
    report_date: Optional[datetime] = None
    created_date: datetime
    key_findings: Optional[Dict[str, Any]] = None
    overall_status: Optional[str] = None
    filename: Optional[str] = None
    
    class Config:
        from_attributes = True


class HistoryListResponse(BaseModel):
    """Response schema for history list."""
    total: int
    page: int
    page_size: int
    items: List[HistoryItemResponse]


# Error Response Schema
class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Health Check Schema
class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    database: bool
    services: Dict[str, bool] = Field(default_factory=dict)
