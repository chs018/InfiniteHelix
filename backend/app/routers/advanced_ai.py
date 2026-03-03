"""
Advanced AI Router - 7 Research-Grade Features
Provides REST API endpoints for all advanced AI capabilities
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import numpy as np
from pathlib import Path

from ..database import get_db
from ..models import Analysis, UploadedFile
from ..routers.auth import get_current_user
from ..services.advanced_ai_service import AdvancedAIService

router = APIRouter(prefix="/api/v1/advanced-ai", tags=["Advanced AI"])

# Initialize service
service = AdvancedAIService()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class DrugInteractionRequest(BaseModel):
    medications: List[str]

class RiskScoreRequest(BaseModel):
    age: int
    gender: str
    total_cholesterol: float
    hdl_cholesterol: float
    systolic_bp: int
    smoking: bool
    diabetes: bool
    on_bp_medication: bool

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

class TrendPredictionRequest(BaseModel):
    metric_name: str
    historical_values: List[float]
    prediction_days: int = 7

class PatientEvaluationRequest(BaseModel):
    age: int
    gender: str
    blood_pressure: Optional[Dict[str, int]] = None
    glucose: Optional[float] = None
    hba1c: Optional[float] = None
    symptoms: Optional[List[str]] = None
    diagnoses: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    potassium: Optional[float] = None


# =============================================================================
# 1. CHEST X-RAY ANALYSIS
# =============================================================================

@router.post("/xray-analysis/{file_id}")
async def analyze_chest_xray(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze chest X-ray using CNN model
    Detects 14 pathologies including pneumonia, cardiomegaly, etc.
    """
    # Get file
    uploaded_file = db.query(UploadedFile).filter(
        UploadedFile.id == file_id,
        UploadedFile.user_id == current_user['id']
    ).first()
    
    if not uploaded_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="X-ray image not found"
        )
    
    # Check if XRay analyzer is available
    if not service.xray_analyzer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="X-ray analysis model not available. Install PyTorch for GPU acceleration."
        )
    
    try:
        # Analyze X-ray
        result = service.xray_analyzer.analyze_xray(uploaded_file.file_path, service.device)
        
        # Save analysis
        analysis = Analysis(
            file_id=file_id,
            analysis_type='xray_cnn',
            status='completed',
            result_data=json.dumps(result)
        )
        db.add(analysis)
        db.commit()
        
        return {
            'success': True,
            'analysis_id': analysis.id,
            'results': result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"X-ray analysis failed: {str(e)}"
        )


# =============================================================================
# 2. ECG INTERPRETATION
# =============================================================================

@router.post("/ecg-interpretation")
async def interpret_ecg(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Interpret ECG signal using 1D CNN
    Detects arrhythmias, MI, and other cardiac conditions
    """
    if not service.ecg_interpreter:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ECG interpreter not available"
        )
    
    try:
        # Read ECG signal (assume CSV format with single column)
        content = await file.read()
        signal_data = content.decode('utf-8').strip().split('\n')
        signal = np.array([float(x) for x in signal_data if x.strip()])
        
        # Interpret ECG
        result = service.ecg_interpreter.interpret_ecg(signal, service.device)
        
        return {
            'success': True,
            'results': result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ECG interpretation failed: {str(e)}"
        )


# =============================================================================
# 3. DRUG INTERACTION CHECK
# =============================================================================

@router.post("/drug-interactions")
async def check_drug_interactions(
    request: DrugInteractionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Check for drug-drug interactions
    Returns severity, risks, and recommendations
    """
    try:
        result = service.drug_predictor.check_interactions(request.medications)
        
        return {
            'success': True,
            'results': result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drug interaction check failed: {str(e)}"
        )


# =============================================================================
# 4. DISEASE RISK SCORING
# =============================================================================

@router.post("/risk-score/cardiovascular")
async def calculate_cvd_risk(
    request: RiskScoreRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Calculate 10-year cardiovascular disease risk using Framingham score
    """
    try:
        result = service.risk_scorer.calculate_framingham_risk(
            age=request.age,
            gender=request.gender,
            total_chol=request.total_cholesterol,
            hdl=request.hdl_cholesterol,
            sbp=request.systolic_bp,
            smoking=request.smoking,
            diabetes=request.diabetes,
            on_bp_meds=request.on_bp_medication
        )
        
        return {
            'success': True,
            'risk_type': 'Framingham 10-Year CVD Risk',
            'results': result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk calculation failed: {str(e)}"
        )


# =============================================================================
# 5. MEDICAL TRANSLATION
# =============================================================================

@router.post("/translate")
async def translate_medical_text(
    request: TranslationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Translate medical text between languages
    Preserves medical terminology accuracy
    """
    try:
        result = service.translator.translate(
            text=request.text,
            source_lang=request.source_language,
            target_lang=request.target_language
        )
        
        return {
            'success': True,
            'results': result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@router.get("/translate/languages")
async def get_supported_languages():
    """Get list of supported languages for translation"""
    return {
        'success': True,
        'languages': service.translator.supported_languages
    }


# =============================================================================
# 6. HEALTH TREND PREDICTION
# =============================================================================

@router.post("/predict-trend")
async def predict_health_trend(
    request: TrendPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Predict future health metric trends using LSTM
    Useful for glucose, BP, weight tracking
    """
    if not service.trend_predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trend prediction model not available"
        )
    
    try:
        result = service.trend_predictor.predict_trend(
            historical_data=request.historical_values,
            steps_ahead=request.prediction_days
        )
        
        return {
            'success': True,
            'metric': request.metric_name,
            'results': result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trend prediction failed: {str(e)}"
        )


# =============================================================================
# 7. CLINICAL DECISION SUPPORT
# =============================================================================

@router.post("/clinical-decision-support")
async def evaluate_patient_cdss(
    request: PatientEvaluationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Comprehensive clinical decision support
    Provides risk assessments, diagnostic suggestions, and treatment recommendations
    """
    try:
        patient_data = request.dict(exclude_none=True)
        result = service.cdss.evaluate_patient(patient_data)
        
        return {
            'success': True,
            'results': result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CDSS evaluation failed: {str(e)}"
        )


# =============================================================================
# SYSTEM STATUS
# =============================================================================

@router.get("/capabilities")
async def get_ai_capabilities():
    """
    Get available AI capabilities and system status
    """
    capabilities = service.get_capabilities()
    
    return {
        'success': True,
        'capabilities': capabilities,
        'device': service.device,
        'features': {
            'chest_xray': 'CNN-based pathology detection (14 conditions)',
            'ecg_interpretation': '1D CNN for arrhythmia detection',
            'drug_interactions': 'Comprehensive drug-drug interaction database',
            'risk_scoring': 'Framingham CVD risk calculator',
            'translation': 'Multi-language medical translation (9 languages)',
            'trend_prediction': 'LSTM-based health metric forecasting',
            'cdss': 'Clinical decision support with evidence-based guidelines'
        }
    }


@router.get("/health")
async def health_check():
    """Health check for advanced AI service"""
    return {
        'status': 'healthy',
        'service': 'Advanced AI',
        'gpu_available': service.device == 'cuda',
        'models_loaded': sum(service.get_capabilities().values())
    }
