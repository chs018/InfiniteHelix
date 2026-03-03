"""
Analyze router for medical report analysis endpoints.
Handles OCR processing and NLP analysis of uploaded files.
"""

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from sqlalchemy.orm import Session
import logging
import time
from typing import Optional

from app.database import get_db
from app.schemas import AnalysisRequest, AnalysisResponse
from app.models import UploadedFile, Analysis, MedicalMetric, HealthInsight
from app.services.ocr_service import ocr_service
from app.services.nlp_service import nlp_service
from app.services.medical_service import medical_service
from app.services.ml_service import ml_service  # Import ML service
from app.services.groq_agent_service import groq_agent  # Import Groq agent
from app.utils.validators import validate_file_id, validate_medical_value
from app.routers.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyze"])


async def process_analysis(
    file_id: str,
    analysis_id: str,
    db: Session
):
    """
    Background task to process file analysis.
    
    Args:
        file_id: ID of the file to analyze
        analysis_id: ID of the analysis record
        db: Database session
    """
    try:
        start_time = time.time()
        
        # Get file and analysis records
        db_file = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
        db_analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        
        if not db_file or not db_analysis:
            logger.error(f"File or analysis not found: {file_id}, {analysis_id}")
            return
        
        # Update status
        db_analysis.status = "processing"
        db.commit()
        
        # Step 1: Extract text using OCR
        logger.info(f"Starting OCR for file: {file_id}")
        extracted_text, ocr_confidence = ocr_service.extract_text_from_file(
            db_file.file_path,
            db_file.file_type
        )
        
        if not extracted_text:
            raise Exception("No text could be extracted from the file")
        
        db_analysis.extracted_text = extracted_text
        db_analysis.ocr_confidence = ocr_confidence
        db.commit()
        
        # Step 2: Analyze text with NLP (traditional spaCy)
        logger.info(f"Starting NLP analysis for file: {file_id}")
        nlp_results = nlp_service.analyze_text(extracted_text)
        
        db_analysis.entities = nlp_results.get('entities', {})
        db_analysis.keywords = nlp_results.get('keywords', [])
        db.commit()
        
        # Step 2.5: Advanced ML Analysis (BioBERT + Medical AI)
        logger.info(f"Starting ML-powered analysis for file: {file_id}")
        ml_extracted_data = {}
        
        # Extract medical entities using BioBERT
        try:
            ml_entities = ml_service.extract_medical_entities(extracted_text)
            logger.info(f"ML extracted {sum(len(v) for v in ml_entities.values())} medical entities")
        except Exception as ml_error:
            logger.warning(f"ML entity extraction failed: {ml_error}")
            ml_entities = {}
        
        # Analyze blood pressure
        try:
            bp_analysis = ml_service.analyze_blood_pressure(extracted_text)
            if bp_analysis['readings']:
                ml_extracted_data['blood_pressure'] = bp_analysis
                logger.info(f"Detected {len(bp_analysis['readings'])} BP reading(s)")
        except Exception as bp_error:
            logger.warning(f"BP analysis failed: {bp_error}")
        
        # Analyze blood sugar/glucose
        try:
            glucose_analysis = ml_service.analyze_blood_sugar(extracted_text)
            if glucose_analysis['readings']:
                ml_extracted_data['blood_sugar'] = glucose_analysis
                logger.info(f"Detected {len(glucose_analysis['readings'])} glucose reading(s)")
        except Exception as glucose_error:
            logger.warning(f"Glucose analysis failed: {glucose_error}")
        
        # Analyze medications
        try:
            medications = ml_service.analyze_medication(extracted_text)
            if medications:
                ml_extracted_data['medications'] = medications
                logger.info(f"Detected {len(medications)} medication(s)")
        except Exception as med_error:
            logger.warning(f"Medication analysis failed: {med_error}")
        
        # Step 3: Extract and assess medical values (traditional method)
        logger.info(f"Extracting medical values for file: {file_id}")
        medical_values = nlp_results.get('medical_values', [])
        
        metrics_list = []
        for value_data in medical_values:
            test_name = value_data.get('test_name', '')
            value_str = value_data.get('value', '')
            unit = value_data.get('unit', '')
            
            # Convert value to float
            value_float = validate_medical_value(value_str)
            if value_float is None:
                continue
            
            # Get reference range and assess value
            ref_range = medical_service.get_reference_range(test_name)
            
            if ref_range:
                status_val, severity, explanation = medical_service.assess_value(
                    test_name, value_float, unit
                )
                
                # Create medical metric
                metric = MedicalMetric(
                    analysis_id=analysis_id,
                    metric_name=test_name,
                    metric_value=value_str,
                    metric_unit=unit,
                    reference_min=ref_range.min_value,
                    reference_max=ref_range.max_value,
                    reference_range=f"{ref_range.min_value}-{ref_range.max_value} {ref_range.unit}",
                    status=status_val,
                    severity=severity,
                    category=nlp_results.get('category', 'general'),
                    notes=explanation
                )
                
                db.add(metric)
                metrics_list.append({
                    'metric_name': test_name,
                    'metric_value': value_str,
                    'status': status_val,
                    'severity': severity
                })
        
        # Add ML-detected BP readings to metrics
        if 'blood_pressure' in ml_extracted_data:
            for bp_reading in ml_extracted_data['blood_pressure']['readings']:
                metric = MedicalMetric(
                    analysis_id=analysis_id,
                    metric_name='Blood Pressure',
                    metric_value=bp_reading['reading'],
                    metric_unit='mmHg',
                    reference_min=90.0,
                    reference_max=120.0,
                    reference_range='90-120 / 60-80 mmHg',
                    status=bp_reading['classification'],
                    severity=bp_reading['risk_level'],
                    category='cardiovascular',
                    notes=f"Systolic: {bp_reading['systolic']}, Diastolic: {bp_reading['diastolic']}"
                )
                db.add(metric)
        
        # Add ML-detected glucose readings to metrics
        if 'blood_sugar' in ml_extracted_data:
            for glucose_reading in ml_extracted_data['blood_sugar']['readings']:
                metric = MedicalMetric(
                    analysis_id=analysis_id,
                    metric_name=f"{glucose_reading['type'].title()} Glucose",
                    metric_value=str(glucose_reading['value']),
                    metric_unit=glucose_reading['unit'],
                    reference_min=70.0 if glucose_reading['type'] == 'fasting' else 70.0,
                    reference_max=100.0 if glucose_reading['type'] == 'fasting' else 140.0,
                    reference_range=f"{'70-100' if glucose_reading['type'] == 'fasting' else '70-140'} {glucose_reading['unit']}",
                    status=glucose_reading['classification'],
                    severity=glucose_reading['risk'],
                    category='metabolic',
                    notes=f"Type: {glucose_reading['type']}, Risk: {glucose_reading['risk']}"
                )
                db.add(metric)
        
        db.commit()
        
        # Step 4: Generate health insights (traditional + ML)
        logger.info(f"Generating health insights for file: {file_id}")
        
        # Traditional insights
        insights = medical_service.generate_insights(metrics_list)
        
        # ML-powered insights
        if ml_extracted_data:
            ml_insights = ml_service.generate_health_insights(ml_extracted_data)
            insights.extend(ml_insights)
            logger.info(f"Generated {len(ml_insights)} ML-powered insights")
        
        for insight_data in insights:
            insight = HealthInsight(
                analysis_id=analysis_id,
                insight_type=insight_data.get('type', 'general'),
                title=insight_data['title'],
                description=insight_data['description'],
                severity=insight_data.get('severity', 'info'),
                priority=insight_data.get('priority', 0),
                is_actionable=insight_data.get('is_actionable', False)
            )
            db.add(insight)
        
        db.commit()
        
        # Step 5: Groq AI Agent Analysis (Advanced Reasoning)
        if groq_agent.is_available():
            logger.info(f"Starting Groq AI agent analysis for file: {file_id}")
            try:
                groq_analysis = await groq_agent.analyze_medical_report(
                    extracted_text=extracted_text,
                    detected_metrics=metrics_list,
                    ml_entities=ml_entities if 'ml_entities' in locals() else {},
                    bp_data=ml_extracted_data.get('blood_pressure'),
                    glucose_data=ml_extracted_data.get('blood_sugar'),
                    medications=ml_extracted_data.get('medications')
                )
                
                # Add Groq-generated insights
                if groq_analysis.get('summary'):
                    insight = HealthInsight(
                        analysis_id=analysis_id,
                        insight_type='ai_summary',
                        title='AI-Generated Medical Summary',
                        description=groq_analysis['summary'],
                        severity='info',
                        priority=400,  # High priority for AI summary
                        is_actionable=False
                    )
                    db.add(insight)
                
                # Add risk assessments
                for risk in groq_analysis.get('risk_assessment', []):
                    severity_map = {
                        'low': 'info',
                        'moderate': 'info',
                        'high': 'warning',
                        'critical': 'critical'
                    }
                    priority_map = {
                        'low': 150,
                        'moderate': 250,
                        'high': 350,
                        'critical': 600
                    }
                    
                    insight = HealthInsight(
                        analysis_id=analysis_id,
                        insight_type='risk_assessment',
                        title=f"Risk: {risk['risk_name']}",
                        description=f"{risk['explanation']} - {risk['primary_concern']}",
                        severity=severity_map.get(risk['risk_level'], 'info'),
                        priority=priority_map.get(risk['risk_level'], 200),
                        is_actionable=risk['risk_level'] in ['high', 'critical']
                    )
                    db.add(insight)
                
                # Add clinical insights
                for clinical in groq_analysis.get('clinical_insights', []):
                    insight = HealthInsight(
                        analysis_id=analysis_id,
                        insight_type='clinical_insight',
                        title=f"Clinical: {clinical['observation'][:50]}...",
                        description=f"{clinical['observation']} | Significance: {clinical['significance']} | Implications: {clinical['implications']}",
                        severity='info',
                        priority=280,
                        is_actionable=True
                    )
                    db.add(insight)
                
                # Add AI recommendations
                for rec in groq_analysis.get('recommendations', []):
                    priority_map = {
                        'low': 120,
                        'medium': 220,
                        'high': 320,
                        'urgent': 550
                    }
                    severity_map = {
                        'low': 'info',
                        'medium': 'info',
                        'high': 'warning',
                        'urgent': 'critical'
                    }
                    
                    insight = HealthInsight(
                        analysis_id=analysis_id,
                        insight_type=f"recommendation_{rec['category']}",
                        title=f"{rec['category'].title()}: {rec['recommendation'][:60]}...",
                        description=f"{rec['recommendation']} | Rationale: {rec['rationale']}",
                        severity=severity_map.get(rec['priority'], 'info'),
                        priority=priority_map.get(rec['priority'], 200),
                        is_actionable=True
                    )
                    db.add(insight)
                
                # Add red flags (critical warnings)
                for flag in groq_analysis.get('red_flags', []):
                    insight = HealthInsight(
                        analysis_id=analysis_id,
                        insight_type='red_flag',
                        title=f"⚠️ CRITICAL: {flag['flag']}",
                        description=f"Urgency: {flag['urgency'].upper()} | Required Action: {flag['action']}",
                        severity='critical',
                        priority=900,  # Highest priority
                        is_actionable=True
                    )
                    db.add(insight)
                
                # Add patient education points
                education_points = groq_analysis.get('patient_education', [])
                if education_points:
                    education_text = "\n".join([f"• {point}" for point in education_points[:5]])
                    insight = HealthInsight(
                        analysis_id=analysis_id,
                        insight_type='patient_education',
                        title='Understanding Your Health Report',
                        description=education_text,
                        severity='info',
                        priority=100,
                        is_actionable=False
                    )
                    db.add(insight)
                
                # Add follow-up plan
                follow_up = groq_analysis.get('follow_up_plan', {})
                if follow_up:
                    follow_up_text = f"Next Visit: {follow_up.get('next_visit_timeframe', 'As advised')}\n"
                    follow_up_text += f"Monitoring: {follow_up.get('monitoring_frequency', 'Regular')}\n"
                    if follow_up.get('tests_needed'):
                        follow_up_text += f"Tests: {', '.join(follow_up['tests_needed'])}\n"
                    if follow_up.get('specialist_referrals'):
                        follow_up_text += f"Referrals: {', '.join(follow_up['specialist_referrals'])}"
                    
                    insight = HealthInsight(
                        analysis_id=analysis_id,
                        insight_type='follow_up_plan',
                        title='Follow-Up Care Plan',
                        description=follow_up_text,
                        severity='info',
                        priority=380,
                        is_actionable=True
                    )
                    db.add(insight)
                
                db.commit()
                logger.info(f"Groq AI analysis completed - added {len(groq_analysis.get('risk_assessment', [])) + len(groq_analysis.get('recommendations', []))} AI insights")
                
            except Exception as groq_error:
                logger.error(f"Groq agent analysis failed: {groq_error}")
                # Continue even if Groq fails
        else:
            logger.info("Groq agent not available - skipping AI reasoning")
        
        # Update analysis status
        processing_time = time.time() - start_time
        db_analysis.status = "completed"
        db_analysis.processing_time = processing_time
        db.commit()
        
        logger.info(f"Analysis completed for file: {file_id} in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error processing analysis: {str(e)}")
        
        # Update analysis with error
        try:
            db_analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            if db_analysis:
                db_analysis.status = "failed"
                db_analysis.error_message = str(e)
                db.commit()
        except:
            pass


@router.post(
    "/analyze/{file_id}",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze medical report",
    description="Start analysis of an uploaded medical report file"
)
async def analyze_file(
    file_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Analyze an uploaded medical report file.
    
    This endpoint triggers OCR processing, NLP analysis, and medical interpretation.
    Processing happens in the background. Use the returned analysis_id to check results.
    """
    try:
        # Validate file ID
        validate_file_id(file_id)
        
        # Check if file exists
        db_file = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
        
        if not db_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Check if file is already being analyzed
        existing_analysis = db.query(Analysis).filter(
            Analysis.file_id == file_id,
            Analysis.status.in_(["pending", "processing"])
        ).first()
        
        if existing_analysis:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="File is already being analyzed"
            )
        
        # Create analysis record
        analysis = Analysis(
            file_id=file_id,
            status="pending"
        )
        
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        # Start background processing
        background_tasks.add_task(process_analysis, file_id, analysis.id, db)
        
        logger.info(f"Analysis started for file: {file_id}")
        
        # Return initial response
        return AnalysisResponse(
            id=analysis.id,
            file_id=analysis.file_id,
            analysis_date=analysis.analysis_date,
            status=analysis.status,
            extracted_text=None,
            ocr_confidence=None,
            entities=None,
            keywords=None,
            processing_time=None,
            error_message=None,
            metrics=[],
            insights=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while starting the analysis"
        )


@router.get(
    "/analyze/{analysis_id}/status",
    summary="Get analysis status",
    description="Check the status of an ongoing analysis"
)
async def get_analysis_status(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get the current status of an analysis by analysis ID.
    """
    try:
        # Validate analysis ID
        validate_file_id(analysis_id)  # Reusing same UUID validator
        
        # Get analysis by ID
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        return {
            "analysis_id": analysis.id,
            "file_id": analysis.file_id,
            "status": analysis.status,
            "analysis_date": analysis.analysis_date,
            "processing_time": analysis.processing_time,
            "error_message": analysis.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the analysis status"
        )
