"""
Results router for retrieving analysis results and history.
Provides access to medical analysis results and user history.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging
from typing import Optional

from app.database import get_db
from app.schemas import (
    AnalysisResponse,
    MedicalMetricResponse,
    HealthInsightResponse,
    HistoryListResponse,
    HistoryItemResponse
)
from app.models import Analysis, UploadedFile, AnalysisHistory, MedicalMetric, HealthInsight
from app.utils.validators import validate_analysis_id, validate_pagination
from app.routers.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["results"])


@router.get(
    "/results/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get analysis results",
    description="Retrieve complete analysis results including metrics and insights"
)
async def get_analysis_results(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get detailed analysis results for a specific analysis.
    
    Returns complete information including extracted text, medical metrics,
    and AI-generated health insights.
    """
    try:
        # Validate analysis ID
        validate_analysis_id(analysis_id)
        
        # Get analysis with relationships
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Check if analysis is complete
        if analysis.status == "pending":
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="Analysis is still pending. Please check back later."
            )
        
        if analysis.status == "processing":
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="Analysis is still in progress. Please check back later."
            )
        
        if analysis.status == "failed":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {analysis.error_message}"
            )
        
        # Build response
        return AnalysisResponse(
            id=analysis.id,
            file_id=analysis.file_id,
            analysis_date=analysis.analysis_date,
            extracted_text=analysis.extracted_text,
            ocr_confidence=analysis.ocr_confidence,
            entities=analysis.entities,
            keywords=analysis.keywords,
            status=analysis.status,
            processing_time=analysis.processing_time,
            error_message=analysis.error_message,
            metrics=[
                MedicalMetricResponse(
                    id=metric.id,
                    analysis_id=metric.analysis_id,
                    metric_name=metric.metric_name,
                    metric_value=metric.metric_value,
                    metric_unit=metric.metric_unit,
                    reference_min=metric.reference_min,
                    reference_max=metric.reference_max,
                    reference_range=metric.reference_range,
                    status=metric.status,
                    severity=metric.severity,
                    category=metric.category,
                    notes=metric.notes
                )
                for metric in analysis.metrics
            ],
            insights=[
                HealthInsightResponse(
                    id=insight.id,
                    analysis_id=insight.analysis_id,
                    insight_type=insight.insight_type,
                    title=insight.title,
                    description=insight.description,
                    severity=insight.severity,
                    priority=insight.priority,
                    is_actionable=insight.is_actionable,
                    created_date=insight.created_date
                )
                for insight in sorted(analysis.insights, key=lambda x: x.priority, reverse=True)
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the analysis results"
        )


@router.get(
    "/results/{analysis_id}/metrics",
    response_model=list[MedicalMetricResponse],
    summary="Get medical metrics",
    description="Retrieve only the medical metrics from an analysis"
)
async def get_analysis_metrics(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get medical metrics for a specific analysis.
    """
    try:
        validate_analysis_id(analysis_id)
        
        # Check if analysis exists
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Get metrics
        metrics = db.query(MedicalMetric).filter(
            MedicalMetric.analysis_id == analysis_id
        ).all()
        
        return [
            MedicalMetricResponse(
                id=metric.id,
                analysis_id=metric.analysis_id,
                metric_name=metric.metric_name,
                metric_value=metric.metric_value,
                metric_unit=metric.metric_unit,
                reference_min=metric.reference_min,
                reference_max=metric.reference_max,
                reference_range=metric.reference_range,
                status=metric.status,
                severity=metric.severity,
                category=metric.category,
                notes=metric.notes
            )
            for metric in metrics
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the metrics"
        )


@router.get(
    "/results/{analysis_id}/insights",
    response_model=list[HealthInsightResponse],
    summary="Get health insights",
    description="Retrieve only the health insights from an analysis"
)
async def get_analysis_insights(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get health insights for a specific analysis.
    """
    try:
        validate_analysis_id(analysis_id)
        
        # Check if analysis exists
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Get insights ordered by priority
        insights = db.query(HealthInsight).filter(
            HealthInsight.analysis_id == analysis_id
        ).order_by(desc(HealthInsight.priority)).all()
        
        return [
            HealthInsightResponse(
                id=insight.id,
                analysis_id=insight.analysis_id,
                insight_type=insight.insight_type,
                title=insight.title,
                description=insight.description,
                severity=insight.severity,
                priority=insight.priority,
                is_actionable=insight.is_actionable,
                created_date=insight.created_date
            )
            for insight in insights
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving insights: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the insights"
        )


@router.get(
    "/history",
    response_model=HistoryListResponse,
    summary="Get analysis history",
    description="Retrieve user's analysis history with pagination"
)
async def get_analysis_history(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    user_id: Optional[str] = Query(None, description="User ID for filtering"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get paginated list of previous analyses.
    """
    try:
        # Validate pagination
        page, page_size = validate_pagination(page, page_size)
        
        # Build query
        query = db.query(Analysis).join(UploadedFile)
        
        # Filter by user if provided
        if user_id:
            query = query.filter(UploadedFile.user_id == user_id)
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        analyses = query.order_by(desc(Analysis.analysis_date)).offset(
            (page - 1) * page_size
        ).limit(page_size).all()
        
        # Build response items
        items = []
        for analysis in analyses:
            # Get key findings
            metrics = db.query(MedicalMetric).filter(
                MedicalMetric.analysis_id == analysis.id,
                MedicalMetric.status.in_(['high', 'low', 'critical'])
            ).limit(5).all()
            
            key_findings = {
                'abnormal_count': len(metrics),
                'metrics': [
                    {
                        'name': m.metric_name,
                        'status': m.status
                    }
                    for m in metrics
                ]
            }
            
            # Determine overall status
            if analysis.status != 'completed':
                overall_status = analysis.status
            elif not metrics:
                overall_status = 'good'
            elif any(m.severity == 'severe' for m in metrics):
                overall_status = 'critical'
            else:
                overall_status = 'attention_needed'
            
            items.append(
                HistoryItemResponse(
                    id=0,  # Placeholder, will be from AnalysisHistory in production
                    analysis_id=analysis.id,
                    report_type=analysis.entities.get('category') if analysis.entities else None,
                    report_date=analysis.analysis_date,
                    created_date=analysis.analysis_date,
                    key_findings=key_findings,
                    overall_status=overall_status,
                    filename=analysis.file.original_filename if analysis.file else None
                )
            )
        
        return HistoryListResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=items
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the history"
        )
