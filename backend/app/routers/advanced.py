"""
Advanced ML Features Router
Fine-tuned BioBERT, Custom NER, Anomaly Detection, Longitudinal Tracking
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging

from app.database import get_db
from app.models import Analysis, MedicalMetric, HealthInsight, UploadedFile
from app.routers.auth import get_current_user
from app.services.advanced_ml_service import advanced_ml_service
from pydantic import BaseModel

router = APIRouter(
    prefix="/api/v1/advanced",
    tags=["advanced-ml"]
)

logger = logging.getLogger(__name__)


# ===== REQUEST/RESPONSE MODELS =====

class AdvancedEntityExtractionResponse(BaseModel):
    """Response for advanced entity extraction"""
    analysis_id: str
    entities: Dict[str, List[Dict]]
    entity_count: int
    message: str


class MedicationExtractionResponse(BaseModel):
    """Response for medication extraction"""
    analysis_id: str
    medications: List[Dict[str, Any]]
    medication_count: int
    message: str


class AnomalyDetectionResponse(BaseModel):
    """Response for anomaly detection"""
    analysis_id: str
    anomalies: List[Dict[str, Any]]
    anomaly_count: int
    total_metrics: int
    message: str


class LongitudinalAnalysisResponse(BaseModel):
    """Response for longitudinal analysis"""
    user_id: str
    trends: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]
    risk_changes: List[Dict[str, Any]]
    insights: List[str]
    data_points: int
    tracking_period_days: int
    message: str


# ===== ENDPOINTS =====

@router.post("/entity-extraction/{analysis_id}", response_model=AdvancedEntityExtractionResponse)
async def extract_entities_advanced(
    analysis_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Extract medical entities using fine-tuned BioBERT
    More accurate than base model - detects diseases, chemicals, genes, proteins
    """
    try:
        # Get analysis
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Get extracted text
        text = analysis.extracted_text
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No extracted text available"
            )
        
        # Extract entities with fine-tuned BioBERT
        entities = advanced_ml_service.extract_medical_entities_advanced(text)
        
        entity_count = sum(len(v) for v in entities.values())
        
        return AdvancedEntityExtractionResponse(
            analysis_id=analysis_id,
            entities=entities,
            entity_count=entity_count,
            message=f"Extracted {entity_count} entities using fine-tuned BioBERT"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced entity extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract entities: {str(e)}"
        )


@router.post("/medication-extraction/{analysis_id}", response_model=MedicationExtractionResponse)
async def extract_medications_advanced(
    analysis_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Extract medications using custom NER model
    Detects drug names, dosages, and frequencies
    """
    try:
        # Get analysis
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Get extracted text
        text = analysis.extracted_text
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No extracted text available"
            )
        
        # Extract medications with custom NER
        medications = advanced_ml_service.extract_medications_advanced(text)
        
        return MedicationExtractionResponse(
            analysis_id=analysis_id,
            medications=medications,
            medication_count=len(medications),
            message=f"Extracted {len(medications)} medications with custom NER model"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in medication extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract medications: {str(e)}"
        )


@router.post("/anomaly-detection/{analysis_id}", response_model=AnomalyDetectionResponse)
async def detect_lab_anomalies(
    analysis_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Detect anomalous lab results using Isolation Forest
    Identifies unusual patterns in medical metrics
    """
    try:
        # Get analysis
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Get all metrics for this analysis
        metrics = db.query(MedicalMetric).filter(
            MedicalMetric.analysis_id == analysis_id
        ).all()
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No metrics available for anomaly detection"
            )
        
        # Convert to dict format
        metrics_data = [
            {
                'metric_name': m.metric_name,
                'metric_value': m.metric_value,
                'reference_min': m.reference_min,
                'reference_max': m.reference_max,
                'unit': m.metric_unit,
                'status': m.status
            }
            for m in metrics
        ]
        
        # Detect anomalies
        anomalies = advanced_ml_service.detect_anomalies(metrics_data)
        
        return AnomalyDetectionResponse(
            analysis_id=analysis_id,
            anomalies=anomalies,
            anomaly_count=len(anomalies),
            total_metrics=len(metrics_data),
            message=f"Detected {len(anomalies)} anomalous results out of {len(metrics_data)} metrics"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect anomalies: {str(e)}"
        )


@router.get("/longitudinal-analysis", response_model=LongitudinalAnalysisResponse)
async def analyze_longitudinal_health(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze health trends over time using all user's analyses
    Tracks metrics longitudinally and predicts future trends
    """
    try:
        # Get all user's analyses ordered by date
        # Join with UploadedFile to filter by user_id
        analyses = db.query(Analysis).join(
            UploadedFile, Analysis.file_id == UploadedFile.id
        ).filter(
            UploadedFile.user_id == str(current_user['id'])
        ).order_by(Analysis.analysis_date.desc()).all()
        
        if len(analyses) < 2:
            return LongitudinalAnalysisResponse(
                user_id=str(current_user['id']),
                trends=[],
                predictions=[],
                risk_changes=[],
                insights=["Need at least 2 analyses for trend tracking. Upload more reports!"],
                data_points=len(analyses),
                tracking_period_days=0,
                message="Insufficient data for longitudinal analysis"
            )
        
        # Get current (latest) analysis metrics
        latest_analysis = analyses[0]
        current_metrics = db.query(MedicalMetric).filter(
            MedicalMetric.analysis_id == latest_analysis.id
        ).all()
        
        current_metrics_data = [
            {
                'metric_name': m.metric_name,
                'metric_value': m.metric_value,
                'reference_min': m.reference_min,
                'reference_max': m.reference_max,
                'unit': m.unit,
                'status': m.status
            }
            for m in current_metrics
        ]
        
        # Get historical analyses data
        historical_data = []
        for analysis in analyses[1:]:  # Skip latest (current)
            metrics = db.query(MedicalMetric).filter(
                MedicalMetric.analysis_id == analysis.id
            ).all()
            
            historical_data.append({
                'analysis_id': str(analysis.id),
                'analysis_date': analysis.analysis_date,
                'metrics': [
                    {
                        'metric_name': m.metric_name,
                        'metric_value': m.metric_value,
                        'status': m.status
                    }
                    for m in metrics
                ]
            })
        
        # Perform longitudinal analysis
        trend_results = advanced_ml_service.analyze_health_trends(
            current_metrics_data,
            historical_data
        )
        
        return LongitudinalAnalysisResponse(
            user_id=str(current_user['id']),
            trends=trend_results.get('trends', []),
            predictions=trend_results.get('predictions', []),
            risk_changes=trend_results.get('risk_changes', []),
            insights=trend_results.get('insights', []),
            data_points=trend_results.get('data_points', 0),
            tracking_period_days=trend_results.get('tracking_period_days', 0),
            message=f"Analyzed {len(analyses)} reports over {trend_results.get('tracking_period_days', 0)} days"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in longitudinal analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform longitudinal analysis: {str(e)}"
        )


@router.get("/health-check")
async def advanced_ml_health_check():
    """Check status of advanced ML features"""
    try:
        status = {
            'fine_tuned_biobert': advanced_ml_service.biobert_ner is not None,
            'medication_ner': advanced_ml_service.medication_ner is not None,
            'anomaly_detector': advanced_ml_service.anomaly_detector is not None,
            'longitudinal_tracking': True,
            'message': 'Advanced ML service is operational'
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error checking advanced ML health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
