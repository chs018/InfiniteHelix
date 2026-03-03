"""
Groq-Powered Agentic AI Service for Medical Analysis
Ultra-fast LLM reasoning for intelligent health insights
"""
import os
import logging
from typing import Dict, List, Any, Optional
from groq import Groq
import json

logger = logging.getLogger(__name__)


class GroqAgentService:
    """
    Agentic AI service using Groq for medical reasoning and analysis
    """
    
    def __init__(self):
        # Initialize Groq client
        api_key = os.getenv('GROQ_API_KEY', '')
        if not api_key:
            logger.warning("GROQ_API_KEY not set. Groq agent will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
            logger.info("Groq Agent Service initialized successfully")
        
        # Use Llama 3.3 70B or Llama 3.1 8B (faster, still great quality)
        # Updated model - llama-3.1-70b-versatile has been decommissioned
        self.model = "llama-3.3-70b-versatile"  # Or use "llama-3.1-8b-instant" for faster responses
        
    def is_available(self) -> bool:
        """Check if Groq service is available"""
        return self.client is not None
    
    async def analyze_medical_report(
        self,
        extracted_text: str,
        detected_metrics: Dict[str, Any],
        ml_entities: Dict[str, List],
        bp_data: Optional[Dict] = None,
        glucose_data: Optional[Dict] = None,
        medications: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Use Groq LLM to perform deep medical reasoning and generate insights
        """
        if not self.is_available():
            logger.warning("Groq service not available, skipping AI analysis")
            return self._fallback_analysis()
        
        try:
            # Prepare context for the agent
            context = self._prepare_context(
                extracted_text, detected_metrics, ml_entities,
                bp_data, glucose_data, medications
            )
            
            # Agent reasoning pipeline
            analysis = {
                'summary': await self._generate_summary(context),
                'risk_assessment': await self._assess_risks(context),
                'clinical_insights': await self._generate_clinical_insights(context),
                'recommendations': await self._generate_recommendations(context),
                'follow_up_plan': await self._create_follow_up_plan(context),
                'patient_education': await self._generate_patient_education(context),
                'red_flags': await self._identify_red_flags(context)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in Groq agent analysis: {e}")
            return self._fallback_analysis()
    
    def _prepare_context(
        self,
        extracted_text: str,
        detected_metrics: Dict,
        ml_entities: Dict,
        bp_data: Optional[Dict],
        glucose_data: Optional[Dict],
        medications: Optional[List[Dict]]
    ) -> str:
        """Prepare comprehensive context for the AI agent"""
        
        context_parts = ["# Medical Report Analysis Context\n"]
        
        # Add extracted text summary
        if extracted_text:
            text_preview = extracted_text[:1000] if len(extracted_text) > 1000 else extracted_text
            context_parts.append(f"\n## Extracted Text:\n{text_preview}\n")
        
        # Add blood pressure data
        if bp_data and bp_data.get('readings'):
            context_parts.append("\n## Blood Pressure Readings:")
            for reading in bp_data['readings']:
                context_parts.append(
                    f"- {reading['reading']} - {reading['classification']} "
                    f"(Risk: {reading['risk_level']})"
                )
            if bp_data.get('average'):
                avg = bp_data['average']
                context_parts.append(
                    f"\nAverage: {avg['systolic']}/{avg['diastolic']} mmHg "
                    f"({avg['classification']})"
                )
        
        # Add glucose data
        if glucose_data and glucose_data.get('readings'):
            context_parts.append("\n## Glucose Readings:")
            for reading in glucose_data['readings']:
                context_parts.append(
                    f"- {reading['type'].title()}: {reading['value']} {reading['unit']} - "
                    f"{reading['classification']} (Risk: {reading['risk']})"
                )
            if glucose_data.get('diabetes_risk'):
                context_parts.append(f"\nDiabetes Risk: {glucose_data['diabetes_risk']}")
        
        # Add medications
        if medications:
            context_parts.append("\n## Current Medications:")
            for med in medications:
                context_parts.append(
                    f"- {med['drug_name']} {med['dosage']} - {med['frequency']} "
                    f"({med['classification']})"
                )
        
        # Add detected entities
        if ml_entities:
            context_parts.append("\n## Detected Medical Entities:")
            for entity_type, entities in ml_entities.items():
                if entities:
                    context_parts.append(f"\n{entity_type.title()}: {len(entities)} found")
        
        # Add detected metrics
        if detected_metrics:
            context_parts.append("\n## Laboratory Metrics:")
            context_parts.append(str(detected_metrics))
        
        return "\n".join(context_parts)
    
    async def _generate_summary(self, context: str) -> str:
        """Generate comprehensive medical summary"""
        
        prompt = f"""You are an expert medical AI assistant analyzing a patient's medical report.

{context}

Task: Provide a clear, concise medical summary (2-3 sentences) that captures:
1. The primary health status
2. Key findings or concerns
3. Overall condition assessment

Be professional, accurate, and patient-friendly. Focus on the most important information.

Summary:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary. Please review detailed metrics."
    
    async def _assess_risks(self, context: str) -> List[Dict[str, Any]]:
        """Assess health risks using AI reasoning"""
        
        prompt = f"""You are a medical risk assessment AI analyzing patient data.

{context}

Task: Identify and assess health risks based on the medical data.

For each risk, provide:
1. Risk name
2. Risk level (low/moderate/high/critical)
3. Brief explanation (1 sentence)
4. Primary concern

Return ONLY a valid JSON array with this structure:
[
  {{
    "risk_name": "string",
    "risk_level": "low|moderate|high|critical",
    "explanation": "string",
    "primary_concern": "string"
  }}
]

If no significant risks, return: [{{\"risk_name\": \"No significant risks\", \"risk_level\": \"low\", \"explanation\": \"All parameters within normal ranges\", \"primary_concern\": \"Maintain healthy lifestyle\"}}]

JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            risks = json.loads(content)
            return risks if isinstance(risks, list) else []
            
        except Exception as e:
            logger.error(f"Error assessing risks: {e}")
            return [{
                "risk_name": "Assessment unavailable",
                "risk_level": "moderate",
                "explanation": "Unable to complete automated risk assessment",
                "primary_concern": "Manual review recommended"
            }]
    
    async def _generate_clinical_insights(self, context: str) -> List[Dict[str, str]]:
        """Generate clinical insights with medical reasoning"""
        
        prompt = f"""You are a clinical decision support AI analyzing medical data.

{context}

Task: Provide 3-5 clinical insights that would be valuable for healthcare providers.

For each insight:
1. Clinical observation
2. Medical significance
3. Potential implications

Return ONLY valid JSON array:
[
  {{
    "observation": "string",
    "significance": "string",
    "implications": "string"
  }}
]

JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            insights = json.loads(content)
            return insights if isinstance(insights, list) else []
            
        except Exception as e:
            logger.error(f"Error generating clinical insights: {e}")
            return []
    
    async def _generate_recommendations(self, context: str) -> List[Dict[str, Any]]:
        """Generate personalized medical recommendations"""
        
        prompt = f"""You are a medical advisory AI providing evidence-based recommendations.

{context}

Task: Provide 4-6 specific, actionable recommendations for this patient.

Categories: lifestyle, medication, monitoring, consultation

Return ONLY valid JSON:
[
  {{
    "category": "lifestyle|medication|monitoring|consultation",
    "recommendation": "string",
    "priority": "low|medium|high|urgent",
    "rationale": "string"
  }}
]

JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=700
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            recommendations = json.loads(content)
            return recommendations if isinstance(recommendations, list) else []
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def _create_follow_up_plan(self, context: str) -> Dict[str, Any]:
        """Create personalized follow-up care plan"""
        
        prompt = f"""You are a care coordination AI creating follow-up plans.

{context}

Task: Create a structured follow-up plan.

Return ONLY valid JSON:
{{
  "next_visit_timeframe": "string (e.g., '2 weeks', '1 month')",
  "monitoring_frequency": "string",
  "tests_needed": ["test1", "test2"],
  "specialist_referrals": ["specialty1", "specialty2"],
  "key_metrics_to_track": ["metric1", "metric2"]
}}

JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            plan = json.loads(content)
            return plan if isinstance(plan, dict) else {}
            
        except Exception as e:
            logger.error(f"Error creating follow-up plan: {e}")
            return {
                "next_visit_timeframe": "As advised by physician",
                "monitoring_frequency": "Regular monitoring recommended",
                "tests_needed": [],
                "specialist_referrals": [],
                "key_metrics_to_track": []
            }
    
    async def _generate_patient_education(self, context: str) -> List[str]:
        """Generate patient-friendly educational content"""
        
        prompt = f"""You are a patient education AI explaining medical findings in simple terms.

{context}

Task: Provide 4-6 key educational points the patient should understand about their health status.

Use simple, non-technical language. Be encouraging but honest.

Return ONLY valid JSON array of strings:
["point1", "point2", "point3", ...]

JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,  # Slightly higher for more natural language
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            education = json.loads(content)
            return education if isinstance(education, list) else []
            
        except Exception as e:
            logger.error(f"Error generating patient education: {e}")
            return []
    
    async def _identify_red_flags(self, context: str) -> List[Dict[str, str]]:
        """Identify critical warning signs requiring immediate attention"""
        
        prompt = f"""You are a medical triage AI identifying critical warning signs.

{context}

Task: Identify any RED FLAGS that require immediate medical attention.

Be conservative - only flag truly urgent conditions. If none, return empty array.

Return ONLY valid JSON:
[
  {{
    "flag": "string",
    "urgency": "urgent|emergency",
    "action": "string"
  }}
]

If no red flags: []

JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Very low temperature for critical assessments
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            red_flags = json.loads(content)
            return red_flags if isinstance(red_flags, list) else []
            
        except Exception as e:
            logger.error(f"Error identifying red flags: {e}")
            return []
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when Groq is unavailable"""
        return {
            'summary': 'AI analysis unavailable. Please review metrics manually.',
            'risk_assessment': [],
            'clinical_insights': [],
            'recommendations': [],
            'follow_up_plan': {},
            'patient_education': [],
            'red_flags': []
        }
    
    async def generate_comparison_insights(
        self,
        current_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare current results with historical data to identify trends"""
        
        if not self.is_available() or not historical_data:
            return {}
        
        try:
            # Prepare comparison context
            context = f"""Current Data: {json.dumps(current_data, indent=2)}

Historical Data (last {len(historical_data)} reports):
{json.dumps(historical_data, indent=2)}"""

            prompt = f"""You are a medical trend analysis AI.

{context}

Task: Analyze trends and changes over time.

Return ONLY valid JSON:
{{
  "trend_summary": "string",
  "improvements": ["improvement1", "improvement2"],
  "concerns": ["concern1", "concern2"],
  "stability_assessment": "improving|stable|declining|fluctuating"
}}

JSON:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error in comparison insights: {e}")
            return {}


# Create global instance
groq_agent = GroqAgentService()
