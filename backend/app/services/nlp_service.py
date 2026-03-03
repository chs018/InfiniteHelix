"""
NLP Service for medical text analysis.
Uses spaCy for entity extraction and text processing.
"""

import spacy
import re
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from app.config import settings

logger = logging.getLogger(__name__)


class NLPService:
    """Service for natural language processing of medical text."""
    
    def __init__(self):
        """Initialize NLP service with spaCy model."""
        try:
            self.nlp = spacy.load(settings.spacy_model)
            logger.info(f"Loaded spaCy model: {settings.spacy_model}")
        except OSError:
            logger.warning(f"spaCy model {settings.spacy_model} not found. Please install it.")
            self.nlp = None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from medical text.
        
        Args:
            text: Input medical text
            
        Returns:
            Dictionary of entity types and their values
        """
        if not self.nlp:
            return {}
        
        try:
            doc = self.nlp(text)
            
            entities = defaultdict(list)
            for ent in doc.ents:
                entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return dict(entities)
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {}
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of important keywords
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            
            # Extract nouns and proper nouns
            keywords = [
                token.text.lower()
                for token in doc
                if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.text) > 2
            ]
            
            # Count frequency
            keyword_freq = defaultdict(int)
            for keyword in keywords:
                keyword_freq[keyword] += 1
            
            # Sort by frequency and return top N
            sorted_keywords = sorted(
                keyword_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [keyword for keyword, _ in sorted_keywords[:top_n]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def extract_medical_values(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical test values and metrics from text.
        
        Args:
            text: Input medical text
            
        Returns:
            List of extracted medical values with metadata
        """
        medical_values = []
        
        # Common medical test patterns
        patterns = [
            # Pattern: "Test Name: Value Unit"
            r'([A-Za-z][A-Za-z\s]+?):\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/%]+)',
            # Pattern: "Test Name Value Unit"
            r'([A-Za-z][A-Za-z\s]+?)\s+([0-9]+(?:\.[0-9]+)?)\s+([a-zA-Z/%]+)',
            # Pattern: "Test Name = Value Unit"
            r'([A-Za-z][A-Za-z\s]+?)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/%]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                test_name = match.group(1).strip()
                value = match.group(2).strip()
                unit = match.group(3).strip()
                
                # Filter out very short test names (likely noise)
                if len(test_name) > 3:
                    medical_values.append({
                        'test_name': test_name,
                        'value': value,
                        'unit': unit,
                        'context': match.group(0)
                    })
        
        return medical_values
    
    def categorize_text(self, text: str) -> str:
        """
        Categorize the type of medical report.
        
        Args:
            text: Input medical text
            
        Returns:
            Category of the medical report
        """
        text_lower = text.lower()
        
        # Define category keywords
        categories = {
            'blood_test': ['blood', 'hemoglobin', 'glucose', 'cholesterol', 'platelet', 'wbc', 'rbc'],
            'urine_test': ['urine', 'urinalysis', 'creatinine', 'protein', 'ketones'],
            'imaging': ['x-ray', 'ct scan', 'mri', 'ultrasound', 'radiolog'],
            'cardiac': ['ecg', 'ekg', 'heart', 'cardiac', 'cardio'],
            'pathology': ['biopsy', 'pathology', 'tissue', 'histology'],
        }
        
        # Count matches for each category
        category_scores = defaultdict(int)
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    category_scores[category] += 1
        
        # Return category with highest score
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive NLP analysis on medical text.
        
        Args:
            text: Input medical text
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            return {
                'entities': self.extract_entities(text),
                'keywords': self.extract_keywords(text),
                'medical_values': self.extract_medical_values(text),
                'category': self.categorize_text(text),
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {}


# Global NLP service instance
nlp_service = NLPService()
