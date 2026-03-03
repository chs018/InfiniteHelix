"""
Translation router for multi-language support
"""
from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import logging

from app.database import get_db
from app.models import Analysis, HealthInsight, MedicalMetric
from app.routers.auth import get_current_user
from app.services.translation_service import translation_service
from app.utils.language_codes import SUPPORTED_LANGUAGES, get_language_info

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/translate", tags=["translation"])


class TranslateRequest(BaseModel):
    target_language: str
    sections: Optional[List[str]] = ["all"]  # summary, insights, metrics, all


class TranslateResponse(BaseModel):
    analysis_id: str
    language: str
    language_name: str
    language_native: str
    insights: List[dict]
    metrics: List[dict]


@router.get(
    "/languages",
    summary="Get supported languages",
    description="Get list of all supported languages for translation"
)
async def get_supported_languages():
    """
    Get all supported languages with their details
    """
    languages = []
    
    # Indian languages first
    indian_langs = ['hi', 'ta', 'te', 'kn', 'ml', 'bn', 'mr', 'gu', 'pa', 'ur']
    for code in indian_langs:
        if code in SUPPORTED_LANGUAGES:
            lang_info = SUPPORTED_LANGUAGES[code]
            languages.append({
                'code': code,
                'name': lang_info['name'],
                'native': lang_info['native'],
                'rtl': lang_info['rtl'],
                'category': 'Indian Languages'
            })
    
    # Then English
    languages.append({
        'code': 'en',
        'name': 'English',
        'native': 'English',
        'rtl': False,
        'category': 'Default'
    })
    
    # Then other languages
    other_langs = [k for k in SUPPORTED_LANGUAGES.keys() 
                   if k not in indian_langs and k != 'en']
    for code in other_langs:
        lang_info = SUPPORTED_LANGUAGES[code]
        languages.append({
            'code': code,
            'name': lang_info['name'],
            'native': lang_info['native'],
            'rtl': lang_info['rtl'],
            'category': 'Other Languages'
        })
    
    return {
        'total': len(languages),
        'languages': languages
    }


@router.post(
    "/{analysis_id}",
    response_model=TranslateResponse,
    summary="Translate analysis results",
    description="Translate medical analysis results to target language"
)
async def translate_analysis(
    analysis_id: str,
    request: TranslateRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Translate analysis results to specified language
    """
    # Validate language
    if request.target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Language '{request.target_language}' not supported"
        )
    
    # Get analysis
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    # Get insights
    insights = db.query(HealthInsight).filter(
        HealthInsight.analysis_id == analysis_id
    ).order_by(HealthInsight.priority.desc()).all()
    
    # Get metrics
    metrics = db.query(MedicalMetric).filter(
        MedicalMetric.analysis_id == analysis_id
    ).all()
    
    # Convert to dict
    insights_data = [
        {
            'id': i.id,
            'insight_type': i.insight_type,
            'title': i.title,
            'description': i.description,
            'severity': i.severity,
            'priority': i.priority,
            'is_actionable': i.is_actionable
        }
        for i in insights
    ]
    
    metrics_data = [
        {
            'id': m.id,
            'metric_name': m.metric_name,
            'metric_value': m.metric_value,
            'metric_unit': m.metric_unit,
            'reference_range': m.reference_range,
            'status': m.status,
            'severity': m.severity,
            'category': m.category,
            'notes': m.notes
        }
        for m in metrics
    ]
    
    # Translate
    if request.target_language != 'en':
        logger.info(f"Translating analysis {analysis_id} to {request.target_language}")
        
        if "insights" in request.sections or "all" in request.sections:
            insights_data = await translation_service.translate_insights(
                insights_data,
                request.target_language
            )
        
        if "metrics" in request.sections or "all" in request.sections:
            metrics_data = await translation_service.translate_metrics(
                metrics_data,
                request.target_language
            )
    
    lang_info = get_language_info(request.target_language)
    
    return TranslateResponse(
        analysis_id=analysis_id,
        language=request.target_language,
        language_name=lang_info['name'],
        language_native=lang_info['native'],
        insights=insights_data,
        metrics=metrics_data
    )


@router.get(
    "/{analysis_id}/quick",
    summary="Quick translate summary",
    description="Quickly translate just the main summary"
)
async def quick_translate(
    analysis_id: str,
    lang: str = Query(..., description="Target language code"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Quick translation of main summary only
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Language '{lang}' not supported"
        )
    
    # Get main summary insight
    summary_insight = db.query(HealthInsight).filter(
        HealthInsight.analysis_id == analysis_id,
        HealthInsight.insight_type == 'summary'
    ).first()
    
    if not summary_insight:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Summary not found"
        )
    
    if lang == 'en':
        translated_text = summary_insight.description
    else:
        translated_text = await translation_service.translate_summary(
            summary_insight.description,
            lang,
            "health summary"
        )
    
    lang_info = get_language_info(lang)
    
    return {
        'original': summary_insight.description,
        'translated': translated_text,
        'language': lang,
        'language_name': lang_info['name'],
        'language_native': lang_info['native']
    }
