"""
Advanced Machine Learning Service for Medical Analysis
Uses transformer models for deep medical understanding
"""
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)
import re
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MedicalMLService:
    """
    Advanced ML service using transformer models for medical analysis
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ML Service on device: {self.device}")
        
        # Initialize models lazily to save memory
        self._ner_pipeline = None
        self._sentiment_pipeline = None
        self._classification_pipeline = None
        
    def get_ner_pipeline(self):
        """Lazy load medical NER pipeline"""
        if self._ner_pipeline is None:
            try:
                # Using BioBERT for medical NER
                logger.info("Loading medical NER model (BioBERT)...")
                self._ner_pipeline = pipeline(
                    "ner",
                    model="dmis-lab/biobert-base-cased-v1.1",
                    device=0 if self.device == "cuda" else -1,
                    aggregation_strategy="simple"
                )
                logger.info("Medical NER model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading NER model: {e}")
                self._ner_pipeline = None
        return self._ner_pipeline
    
    def get_sentiment_pipeline(self):
        """Lazy load medical sentiment analysis"""
        if self._sentiment_pipeline is None:
            try:
                logger.info("Loading medical sentiment model...")
                self._sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Sentiment model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading sentiment model: {e}")
                self._sentiment_pipeline = None
        return self._sentiment_pipeline
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract medical entities using BioBERT
        Returns diseases, medications, symptoms, procedures, etc.
        """
        try:
            ner_pipeline = self.get_ner_pipeline()
            if not ner_pipeline:
                return self._fallback_entity_extraction(text)
            
            # Process text in chunks (BioBERT has 512 token limit)
            max_length = 500
            chunks = self._chunk_text(text, max_length)
            
            all_entities = []
            for chunk in chunks:
                try:
                    entities = ner_pipeline(chunk)
                    all_entities.extend(entities)
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    continue
            
            # Organize entities by type
            organized = {
                'diseases': [],
                'medications': [],
                'symptoms': [],
                'procedures': [],
                'body_parts': [],
                'test_results': [],
                'other': []
            }
            
            for entity in all_entities:
                entity_data = {
                    'text': entity['word'],
                    'score': float(entity['score']),
                    'type': entity.get('entity_group', 'UNKNOWN')
                }
                
                # Categorize based on entity type and text analysis
                text_lower = entity['word'].lower()
                
                if any(word in text_lower for word in ['diabetes', 'hypertension', 'disease', 'syndrome', 'disorder']):
                    organized['diseases'].append(entity_data)
                elif any(word in text_lower for word in ['tablet', 'mg', 'medication', 'drug', 'capsule', 'syrup']):
                    organized['medications'].append(entity_data)
                elif any(word in text_lower for word in ['pain', 'fever', 'nausea', 'headache', 'fatigue']):
                    organized['symptoms'].append(entity_data)
                elif any(word in text_lower for word in ['surgery', 'procedure', 'operation', 'therapy', 'treatment']):
                    organized['procedures'].append(entity_data)
                elif any(word in text_lower for word in ['heart', 'liver', 'kidney', 'blood', 'brain']):
                    organized['body_parts'].append(entity_data)
                elif any(word in text_lower for word in ['glucose', 'pressure', 'cholesterol', 'hemoglobin', 'count']):
                    organized['test_results'].append(entity_data)
                else:
                    organized['other'].append(entity_data)
            
            # Remove duplicates
            for key in organized:
                organized[key] = self._remove_duplicate_entities(organized[key])
            
            return organized
            
        except Exception as e:
            logger.error(f"Error in ML entity extraction: {e}")
            return self._fallback_entity_extraction(text)
    
    def analyze_medication(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract and analyze medication information including:
        - Drug name
        - Dosage
        - Frequency
        - Duration
        - Route of administration
        """
        medications = []
        
        # Pattern for medication extraction
        # Examples: "Metformin 500mg twice daily", "Aspirin 75 mg once a day"
        med_pattern = r'([A-Z][a-zA-Z]+)\s*(\d+\s*(?:mg|g|ml|mcg|units?))\s*(.*?)(?:\.|,|;|$)'
        
        matches = re.finditer(med_pattern, text, re.IGNORECASE)
        
        for match in matches:
            drug_name = match.group(1).strip()
            dosage = match.group(2).strip()
            instructions = match.group(3).strip()
            
            # Extract frequency
            frequency = self._extract_frequency(instructions)
            duration = self._extract_duration(instructions)
            route = self._extract_route(instructions)
            
            medication = {
                'drug_name': drug_name,
                'dosage': dosage,
                'frequency': frequency,
                'duration': duration,
                'route': route,
                'instructions': instructions,
                'classification': self._classify_medication(drug_name)
            }
            
            medications.append(medication)
        
        return medications
    
    def analyze_blood_pressure(self, text: str) -> Dict[str, Any]:
        """
        Extract and analyze blood pressure readings
        Classify: Normal, Elevated, Stage 1/2 Hypertension, Crisis
        """
        bp_pattern = r'(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*Hg|mmHg)?'
        matches = re.finditer(bp_pattern, text)
        
        bp_readings = []
        for match in matches:
            systolic = int(match.group(1))
            diastolic = int(match.group(2))
            
            # Validate BP range
            if 60 <= systolic <= 250 and 40 <= diastolic <= 150:
                classification = self._classify_blood_pressure(systolic, diastolic)
                risk_level = self._assess_bp_risk(systolic, diastolic)
                
                bp_readings.append({
                    'systolic': systolic,
                    'diastolic': diastolic,
                    'reading': f"{systolic}/{diastolic} mmHg",
                    'classification': classification,
                    'risk_level': risk_level,
                    'recommendations': self._get_bp_recommendations(classification, risk_level)
                })
        
        if bp_readings:
            # Calculate average if multiple readings
            avg_systolic = sum(r['systolic'] for r in bp_readings) / len(bp_readings)
            avg_diastolic = sum(r['diastolic'] for r in bp_readings) / len(bp_readings)
            
            return {
                'readings': bp_readings,
                'average': {
                    'systolic': round(avg_systolic, 1),
                    'diastolic': round(avg_diastolic, 1),
                    'classification': self._classify_blood_pressure(avg_systolic, avg_diastolic)
                },
                'trend': self._analyze_bp_trend(bp_readings) if len(bp_readings) > 1 else None
            }
        
        return {'readings': [], 'average': None, 'trend': None}
    
    def analyze_blood_sugar(self, text: str) -> Dict[str, Any]:
        """
        Extract and analyze blood glucose levels
        Support: Fasting, Random, HbA1c, Post-prandial
        """
        glucose_readings = []
        
        # Fasting glucose pattern
        fasting_pattern = r'(?:fasting|FBS|FPG)\s*:?\s*(\d{2,3})\s*(?:mg/dL|mg/dl)?'
        for match in re.finditer(fasting_pattern, text, re.IGNORECASE):
            value = int(match.group(1))
            if 40 <= value <= 400:
                glucose_readings.append({
                    'type': 'fasting',
                    'value': value,
                    'unit': 'mg/dL',
                    'classification': self._classify_fasting_glucose(value),
                    'risk': self._assess_glucose_risk(value, 'fasting')
                })
        
        # Random glucose pattern
        random_pattern = r'(?:random|RBS|RBG)\s*:?\s*(\d{2,3})\s*(?:mg/dL|mg/dl)?'
        for match in re.finditer(random_pattern, text, re.IGNORECASE):
            value = int(match.group(1))
            if 40 <= value <= 500:
                glucose_readings.append({
                    'type': 'random',
                    'value': value,
                    'unit': 'mg/dL',
                    'classification': self._classify_random_glucose(value),
                    'risk': self._assess_glucose_risk(value, 'random')
                })
        
        # HbA1c pattern
        hba1c_pattern = r'(?:HbA1c|A1C|Hemoglobin A1c)\s*:?\s*(\d+\.?\d*)\s*%?'
        for match in re.finditer(hba1c_pattern, text, re.IGNORECASE):
            value = float(match.group(1))
            if 3.0 <= value <= 15.0:
                glucose_readings.append({
                    'type': 'HbA1c',
                    'value': value,
                    'unit': '%',
                    'classification': self._classify_hba1c(value),
                    'risk': self._assess_hba1c_risk(value),
                    'estimated_avg_glucose': self._convert_hba1c_to_glucose(value)
                })
        
        return {
            'readings': glucose_readings,
            'diabetes_risk': self._calculate_diabetes_risk(glucose_readings),
            'recommendations': self._get_glucose_recommendations(glucose_readings)
        }
    
    def generate_health_insights(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate comprehensive health insights using ML analysis
        """
        insights = []
        
        # Analyze BP data
        if extracted_data.get('blood_pressure'):
            bp_data = extracted_data['blood_pressure']
            if bp_data.get('average'):
                insights.extend(self._generate_bp_insights(bp_data))
        
        # Analyze glucose data
        if extracted_data.get('blood_sugar'):
            glucose_data = extracted_data['blood_sugar']
            insights.extend(self._generate_glucose_insights(glucose_data))
        
        # Analyze medications
        if extracted_data.get('medications'):
            insights.extend(self._generate_medication_insights(extracted_data['medications']))
        
        # Overall health assessment
        insights.append(self._generate_overall_assessment(extracted_data))
        
        # Sort by priority
        insights.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return insights
    
    # Helper methods
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _remove_duplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities"""
        seen = set()
        unique = []
        for entity in entities:
            key = entity['text'].lower()
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique
    
    def _fallback_entity_extraction(self, text: str) -> Dict[str, List]:
        """Fallback entity extraction without ML"""
        return {
            'diseases': [],
            'medications': [],
            'symptoms': [],
            'procedures': [],
            'body_parts': [],
            'test_results': [],
            'other': []
        }
    
    def _extract_frequency(self, instructions: str) -> str:
        """Extract medication frequency"""
        freq_patterns = {
            'once daily': r'(?:once|1\s*time)\s*(?:a\s*day|daily|per\s*day)',
            'twice daily': r'(?:twice|two\s*times?|2\s*times?)\s*(?:a\s*day|daily|per\s*day)',
            'three times daily': r'(?:thrice|three\s*times?|3\s*times?)\s*(?:a\s*day|daily|per\s*day)',
            'four times daily': r'(?:four\s*times?|4\s*times?)\s*(?:a\s*day|daily|per\s*day)',
            'as needed': r'as\s*needed|prn|when\s*required',
            'before meals': r'before\s*(?:meals?|eating|food)',
            'after meals': r'after\s*(?:meals?|eating|food)',
            'at bedtime': r'at\s*(?:bedtime|night|bed)',
        }
        
        for freq, pattern in freq_patterns.items():
            if re.search(pattern, instructions, re.IGNORECASE):
                return freq
        
        return 'as directed'
    
    def _extract_duration(self, instructions: str) -> str:
        """Extract medication duration"""
        duration_pattern = r'for\s*(\d+)\s*(day|week|month|year)s?'
        match = re.search(duration_pattern, instructions, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {match.group(2)}s"
        return 'ongoing'
    
    def _extract_route(self, instructions: str) -> str:
        """Extract route of administration"""
        routes = {
            'oral': r'oral|by\s*mouth|PO|tablet|capsule',
            'intravenous': r'IV|intravenous|into\s*vein',
            'intramuscular': r'IM|intramuscular|into\s*muscle',
            'subcutaneous': r'SC|subcutaneous|under\s*skin',
            'topical': r'topical|apply\s*to\s*skin|cream|ointment',
            'inhaled': r'inhaled|inhalation|nebulizer',
        }
        
        for route, pattern in routes.items():
            if re.search(pattern, instructions, re.IGNORECASE):
                return route
        
        return 'oral'
    
    def _classify_medication(self, drug_name: str) -> str:
        """Classify medication type"""
        drug_classes = {
            'antidiabetic': ['metformin', 'insulin', 'glipizide', 'sitagliptin', 'empagliflozin'],
            'antihypertensive': ['amlodipine', 'losartan', 'atenolol', 'ramipril', 'enalapril'],
            'antibiotic': ['amoxicillin', 'azithromycin', 'ciprofloxacin', 'doxycycline'],
            'analgesic': ['paracetamol', 'ibuprofen', 'aspirin', 'diclofenac'],
            'statin': ['atorvastatin', 'simvastatin', 'rosuvastatin'],
            'anticoagulant': ['warfarin', 'aspirin', 'clopidogrel'],
        }
        
        drug_lower = drug_name.lower()
        for class_name, drugs in drug_classes.items():
            if any(drug in drug_lower for drug in drugs):
                return class_name
        
        return 'other'
    
    def _classify_blood_pressure(self, systolic: float, diastolic: float) -> str:
        """Classify blood pressure reading"""
        if systolic < 120 and diastolic < 80:
            return 'Normal'
        elif 120 <= systolic < 130 and diastolic < 80:
            return 'Elevated'
        elif 130 <= systolic < 140 or 80 <= diastolic < 90:
            return 'Stage 1 Hypertension'
        elif 140 <= systolic < 180 or 90 <= diastolic < 120:
            return 'Stage 2 Hypertension'
        elif systolic >= 180 or diastolic >= 120:
            return 'Hypertensive Crisis'
        else:
            return 'Hypotension'
    
    def _assess_bp_risk(self, systolic: float, diastolic: float) -> str:
        """Assess cardiovascular risk from BP"""
        if systolic >= 180 or diastolic >= 120:
            return 'critical'
        elif systolic >= 140 or diastolic >= 90:
            return 'high'
        elif systolic >= 130 or diastolic >= 80:
            return 'moderate'
        elif systolic >= 120:
            return 'low'
        else:
            return 'normal'
    
    def _get_bp_recommendations(self, classification: str, risk_level: str) -> List[str]:
        """Get BP management recommendations"""
        recommendations = []
        
        if classification == 'Hypertensive Crisis':
            recommendations.append('URGENT: Seek immediate medical attention')
            recommendations.append('This is a medical emergency requiring immediate care')
        elif classification == 'Stage 2 Hypertension':
            recommendations.append('Consult doctor about medication adjustment')
            recommendations.append('Monitor BP daily')
            recommendations.append('Reduce sodium intake to <2g/day')
        elif classification == 'Stage 1 Hypertension':
            recommendations.append('Lifestyle modifications recommended')
            recommendations.append('Regular exercise (150 min/week)')
            recommendations.append('Monitor BP weekly')
        elif classification == 'Elevated':
            recommendations.append('Focus on lifestyle changes')
            recommendations.append('Maintain healthy weight')
            recommendations.append('Limit alcohol and caffeine')
        else:
            recommendations.append('Continue healthy lifestyle')
            recommendations.append('Monitor BP monthly')
        
        return recommendations
    
    def _analyze_bp_trend(self, readings: List[Dict]) -> str:
        """Analyze BP trend"""
        if len(readings) < 2:
            return 'insufficient_data'
        
        systolic_values = [r['systolic'] for r in readings]
        first_half_avg = sum(systolic_values[:len(systolic_values)//2]) / (len(systolic_values)//2)
        second_half_avg = sum(systolic_values[len(systolic_values)//2:]) / (len(systolic_values) - len(systolic_values)//2)
        
        diff = second_half_avg - first_half_avg
        
        if diff > 10:
            return 'increasing'
        elif diff < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def _classify_fasting_glucose(self, value: int) -> str:
        """Classify fasting glucose"""
        if value < 70:
            return 'Hypoglycemia'
        elif value < 100:
            return 'Normal'
        elif value < 126:
            return 'Prediabetes'
        else:
            return 'Diabetes'
    
    def _classify_random_glucose(self, value: int) -> str:
        """Classify random glucose"""
        if value < 70:
            return 'Hypoglycemia'
        elif value < 140:
            return 'Normal'
        elif value < 200:
            return 'Prediabetes'
        else:
            return 'Diabetes'
    
    def _classify_hba1c(self, value: float) -> str:
        """Classify HbA1c"""
        if value < 5.7:
            return 'Normal'
        elif value < 6.5:
            return 'Prediabetes'
        else:
            return 'Diabetes'
    
    def _assess_glucose_risk(self, value: float, test_type: str) -> str:
        """Assess diabetes risk from glucose"""
        if test_type == 'fasting':
            if value >= 126:
                return 'high'
            elif value >= 100:
                return 'moderate'
            elif value < 70:
                return 'critical'
            else:
                return 'normal'
        else:  # random
            if value >= 200:
                return 'high'
            elif value >= 140:
                return 'moderate'
            elif value < 70:
                return 'critical'
            else:
                return 'normal'
    
    def _assess_hba1c_risk(self, value: float) -> str:
        """Assess risk from HbA1c"""
        if value >= 9.0:
            return 'critical'
        elif value >= 7.0:
            return 'high'
        elif value >= 6.5:
            return 'moderate'
        elif value >= 5.7:
            return 'low'
        else:
            return 'normal'
    
    def _convert_hba1c_to_glucose(self, hba1c: float) -> int:
        """Convert HbA1c to estimated average glucose"""
        # Formula: eAG (mg/dL) = 28.7 × HbA1c - 46.7
        return int(28.7 * hba1c - 46.7)
    
    def _calculate_diabetes_risk(self, readings: List[Dict]) -> str:
        """Calculate overall diabetes risk"""
        if not readings:
            return 'unknown'
        
        high_risk = sum(1 for r in readings if r.get('risk') in ['critical', 'high'])
        moderate_risk = sum(1 for r in readings if r.get('risk') == 'moderate')
        
        if high_risk > 0:
            return 'high'
        elif moderate_risk > 0:
            return 'moderate'
        else:
            return 'low'
    
    def _get_glucose_recommendations(self, readings: List[Dict]) -> List[str]:
        """Get glucose management recommendations"""
        if not readings:
            return []
        
        recommendations = []
        risk = self._calculate_diabetes_risk(readings)
        
        if risk == 'high':
            recommendations.append('Consult endocrinologist immediately')
            recommendations.append('Monitor blood glucose 2-4 times daily')
            recommendations.append('Review medication regimen with doctor')
            recommendations.append('Follow strict diabetic diet plan')
        elif risk == 'moderate':
            recommendations.append('Schedule diabetes screening')
            recommendations.append('Adopt low glycemic index diet')
            recommendations.append('Regular exercise (30 min daily)')
            recommendations.append('Monitor glucose weekly')
        else:
            recommendations.append('Maintain healthy lifestyle')
            recommendations.append('Balanced diet with controlled carbohydrates')
            recommendations.append('Regular physical activity')
        
        return recommendations
    
    def _generate_bp_insights(self, bp_data: Dict) -> List[Dict]:
        """Generate BP-specific insights"""
        insights = []
        avg = bp_data['average']
        
        if avg:
            classification = avg['classification']
            priority = 100
            
            if classification == 'Hypertensive Crisis':
                priority = 500
                severity = 'critical'
            elif classification in ['Stage 2 Hypertension', 'Stage 1 Hypertension']:
                priority = 300
                severity = 'warning'
            elif classification == 'Elevated':
                priority = 200
                severity = 'info'
            else:
                priority = 100
                severity = 'success'
            
            insights.append({
                'type': 'blood_pressure',
                'title': f'Blood Pressure: {classification}',
                'description': f"Average reading: {avg['systolic']}/{avg['diastolic']} mmHg. " + 
                              f"Current classification: {classification}",
                'severity': severity,
                'priority': priority,
                'recommendations': bp_data['readings'][0].get('recommendations', []) if bp_data['readings'] else [],
                'is_actionable': severity in ['critical', 'warning']
            })
        
        return insights
    
    def _generate_glucose_insights(self, glucose_data: Dict) -> List[Dict]:
        """Generate glucose-specific insights"""
        insights = []
        readings = glucose_data.get('readings', [])
        
        for reading in readings:
            classification = reading['classification']
            priority = 100
            
            if classification == 'Hypoglycemia' or reading.get('risk') == 'critical':
                priority = 500
                severity = 'critical'
            elif classification == 'Diabetes':
                priority = 300
                severity = 'warning'
            elif classification == 'Prediabetes':
                priority = 200
                severity = 'info'
            else:
                priority = 100
                severity = 'success'
            
            insights.append({
                'type': 'blood_glucose',
                'title': f'{reading["type"].title()} Glucose: {classification}',
                'description': f"Value: {reading['value']} {reading['unit']}. " + 
                              f"Classification: {classification}",
                'severity': severity,
                'priority': priority,
                'recommendations': glucose_data.get('recommendations', []),
                'is_actionable': severity in ['critical', 'warning']
            })
        
        return insights
    
    def _generate_medication_insights(self, medications: List[Dict]) -> List[Dict]:
        """Generate medication-related insights"""
        insights = []
        
        if len(medications) > 5:
            insights.append({
                'type': 'medication',
                'title': 'Multiple Medications Detected',
                'description': f'{len(medications)} medications found. Review for potential drug interactions.',
                'severity': 'info',
                'priority': 150,
                'recommendations': ['Consult pharmacist for drug interaction review'],
                'is_actionable': True
            })
        
        # Check for antidiabetic medications
        antidiabetic = [m for m in medications if m.get('classification') == 'antidiabetic']
        if antidiabetic:
            insights.append({
                'type': 'medication',
                'title': 'Diabetes Management',
                'description': f'{len(antidiabetic)} antidiabetic medication(s) prescribed. ' +
                              'Regular glucose monitoring recommended.',
                'severity': 'info',
                'priority': 180,
                'recommendations': [
                    'Monitor blood glucose as directed',
                    'Watch for signs of hypoglycemia',
                    'Take medications at prescribed times'
                ],
                'is_actionable': True
            })
        
        return insights
    
    def _generate_overall_assessment(self, extracted_data: Dict) -> Dict:
        """Generate overall health assessment"""
        critical_count = 0
        warning_count = 0
        
        # Count critical and warning conditions
        if extracted_data.get('blood_pressure'):
            bp_avg = extracted_data['blood_pressure'].get('average')
            if bp_avg:
                classification = bp_avg['classification']
                if 'Crisis' in classification:
                    critical_count += 1
                elif 'Hypertension' in classification:
                    warning_count += 1
        
        if extracted_data.get('blood_sugar'):
            for reading in extracted_data['blood_sugar'].get('readings', []):
                if reading.get('risk') == 'critical':
                    critical_count += 1
                elif reading.get('classification') in ['Diabetes', 'Prediabetes']:
                    warning_count += 1
        
        # Generate assessment
        if critical_count > 0:
            title = 'Urgent Medical Attention Required'
            description = f'{critical_count} critical condition(s) detected. Seek immediate medical care.'
            severity = 'critical'
            priority = 1000
        elif warning_count > 0:
            title = 'Medical Consultation Recommended'
            description = f'{warning_count} condition(s) requiring medical attention detected.'
            severity = 'warning'
            priority = 250
        else:
            title = 'Health Status: Good'
            description = 'All values within acceptable ranges. Continue maintaining healthy lifestyle.'
            severity = 'success'
            priority = 50
        
        return {
            'type': 'overall_assessment',
            'title': title,
            'description': description,
            'severity': severity,
            'priority': priority,
            'is_actionable': critical_count > 0 or warning_count > 0
        }


# Create global instance
ml_service = MedicalMLService()
