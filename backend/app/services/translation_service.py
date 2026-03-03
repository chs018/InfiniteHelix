"""
Translation service using Groq AI for medical content
"""
import logging
from typing import Dict, Any, List
from groq import Groq
import os
import json

from ..utils.language_codes import SUPPORTED_LANGUAGES, get_language_info

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Medical translation service using Groq LLM
    Ensures medical accuracy while translating
    """
    
    def __init__(self):
        api_key = os.getenv('GROQ_API_KEY', '')
        if not api_key:
            logger.warning("GROQ_API_KEY not set. Translation will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
            logger.info("Translation Service initialized successfully")
        
        self.model = "llama-3.3-70b-versatile"
    
    def is_available(self) -> bool:
        """Check if translation service is available"""
        return self.client is not None
    
    async def translate_summary(
        self,
        text: str,
        target_language: str,
        context: str = "medical report"
    ) -> str:
        """
        Translate medical summary to target language
        """
        if not self.is_available():
            return text
        
        lang_info = get_language_info(target_language)
        
        try:
            prompt = f"""You are a professional medical translator. Translate the following medical {context} to {lang_info['name']} ({lang_info['native']}).

CRITICAL RULES:
1. Maintain medical terminology accuracy
2. Preserve numbers, units, and measurements exactly
3. Keep proper nouns (names, places) unchanged
4. Use culturally appropriate medical language
5. Maintain professional medical tone

Original Text ({context}):
{text}

Translate to {lang_info['name']}. Return ONLY the translation, no explanations."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for accuracy
                max_tokens=2000
            )
            
            translated = response.choices[0].message.content.strip()
            logger.info(f"Translated {context} to {lang_info['name']}")
            return translated
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    async def translate_insights(
        self,
        insights: List[Dict[str, Any]],
        target_language: str
    ) -> List[Dict[str, Any]]:
        """
        Translate list of health insights
        """
        if not self.is_available():
            return insights
        
        translated_insights = []
        
        for insight in insights:
            try:
                translated = {
                    **insight,
                    'title': await self.translate_summary(
                        insight['title'],
                        target_language,
                        "medical insight title"
                    ),
                    'description': await self.translate_summary(
                        insight['description'],
                        target_language,
                        "medical insight description"
                    )
                }
                translated_insights.append(translated)
            except Exception as e:
                logger.error(f"Error translating insight: {e}")
                translated_insights.append(insight)
        
        return translated_insights
    
    async def translate_metrics(
        self,
        metrics: List[Dict[str, Any]],
        target_language: str
    ) -> List[Dict[str, Any]]:
        """
        Translate medical metrics with preserved values
        """
        if not self.is_available():
            return metrics
        
        translated_metrics = []
        
        for metric in metrics:
            try:
                # Translate only the notes, preserve all numerical values
                translated = {
                    **metric,
                    'metric_name': await self.translate_summary(
                        metric.get('metric_name', ''),
                        target_language,
                        "medical metric name"
                    )
                }
                
                if metric.get('notes'):
                    translated['notes'] = await self.translate_summary(
                        metric['notes'],
                        target_language,
                        "medical notes"
                    )
                
                translated_metrics.append(translated)
            except Exception as e:
                logger.error(f"Error translating metric: {e}")
                translated_metrics.append(metric)
        
        return translated_metrics
    
    async def translate_full_analysis(
        self,
        analysis_data: Dict[str, Any],
        target_language: str
    ) -> Dict[str, Any]:
        """
        Translate complete analysis report
        """
        if not self.is_available() or target_language == 'en':
            return analysis_data
        
        lang_info = get_language_info(target_language)
        logger.info(f"Starting full analysis translation to {lang_info['name']}")
        
        try:
            translated = analysis_data.copy()
            
            # Translate insights
            if 'insights' in translated and translated['insights']:
                translated['insights'] = await self.translate_insights(
                    translated['insights'],
                    target_language
                )
            
            # Translate metrics
            if 'metrics' in translated and translated['metrics']:
                translated['metrics'] = await self.translate_metrics(
                    translated['metrics'],
                    target_language
                )
            
            # Add language metadata
            translated['language'] = target_language
            translated['language_name'] = lang_info['name']
            translated['language_native'] = lang_info['native']
            
            logger.info(f"Completed translation to {lang_info['name']}")
            return translated
            
        except Exception as e:
            logger.error(f"Error in full analysis translation: {e}")
            return analysis_data


# Global instance
translation_service = TranslationService()
