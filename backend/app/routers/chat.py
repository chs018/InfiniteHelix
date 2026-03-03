"""
Voice chat router for medical Q&A
"""
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import logging

from app.database import get_db
from app.models import Analysis, HealthInsight, MedicalMetric
from app.routers.auth import get_current_user
from app.services.voice_service import voice_chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["voice-chat"])


class ChatStartRequest(BaseModel):
    analysis_id: str


class ChatStartResponse(BaseModel):
    session_id: str
    message: str
    suggestions: List[str]


class ChatMessageRequest(BaseModel):
    session_id: str
    message: str


class ChatMessageResponse(BaseModel):
    response: str
    timestamp: str


@router.post(
    "/start",
    response_model=ChatStartResponse,
    summary="Start voice chat session",
    description="Start a new voice chat session for medical Q&A"
)
async def start_chat_session(
    request: ChatStartRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Start a new chat session with medical context
    """
    # Get analysis
    analysis = db.query(Analysis).filter(
        Analysis.id == request.analysis_id
    ).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    # Get medical data for context
    insights = db.query(HealthInsight).filter(
        HealthInsight.analysis_id == request.analysis_id
    ).order_by(HealthInsight.priority.desc()).limit(5).all()
    
    metrics = db.query(MedicalMetric).filter(
        MedicalMetric.analysis_id == request.analysis_id
    ).limit(10).all()
    
    medical_context = {
        'summary': insights[0].description if insights else None,
        'metrics': [
            {
                'metric_name': m.metric_name,
                'metric_value': m.metric_value,
                'metric_unit': m.metric_unit,
                'status': m.status
            }
            for m in metrics
        ],
        'insights': [
            {
                'title': i.title,
                'description': i.description
            }
            for i in insights
        ]
    }
    
    # Create session
    session_id = voice_chat_service.create_session(
        current_user.id,
        request.analysis_id
    )
    
    # Get suggested questions
    suggestions = await voice_chat_service.get_suggestions(medical_context)
    
    return ChatStartResponse(
        session_id=session_id,
        message="Hi! I'm your medical assistant. Ask me anything about your report.",
        suggestions=suggestions
    )


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    summary="Send chat message",
    description="Send a message and get AI response"
)
async def send_message(
    request: ChatMessageRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Process user message and return AI response
    """
    # Get session
    session = voice_chat_service.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired"
        )
    
    # Verify user owns this session
    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get medical context
    analysis_id = session['analysis_id']
    
    insights = db.query(HealthInsight).filter(
        HealthInsight.analysis_id == analysis_id
    ).order_by(HealthInsight.priority.desc()).limit(5).all()
    
    metrics = db.query(MedicalMetric).filter(
        MedicalMetric.analysis_id == analysis_id
    ).limit(10).all()
    
    medical_context = {
        'summary': insights[0].description if insights else None,
        'metrics': [
            {
                'metric_name': m.metric_name,
                'metric_value': m.metric_value,
                'metric_unit': m.metric_unit,
                'status': m.status
            }
            for m in metrics
        ],
        'insights': [
            {
                'title': i.title,
                'description': i.description
            }
            for i in insights
        ]
    }
    
    # Get AI response
    response_text = await voice_chat_service.chat(
        request.session_id,
        request.message,
        medical_context
    )
    
    from datetime import datetime
    return ChatMessageResponse(
        response=response_text,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get(
    "/history/{session_id}",
    summary="Get chat history",
    description="Get conversation history for a session"
)
async def get_chat_history(
    session_id: str,
    current_user = Depends(get_current_user)
):
    """
    Get chat history for a session
    """
    session = voice_chat_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return {
        'session_id': session_id,
        'messages': session['messages'],
        'created_at': session['created_at'].isoformat()
    }


@router.get(
    "/status",
    summary="Check voice chat status",
    description="Check if voice chat service is available"
)
async def check_status():
    """
    Check if voice chat is available
    """
    available = voice_chat_service.is_available()
    return {
        'available': available,
        'message': 'Voice chat is ready' if available else 'Voice chat is unavailable'
    }
