"""
Medical Service for interpreting health metrics and generating insights.
Provides medical knowledge and reference ranges for common tests.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReferenceRange:
    """Reference range for a medical test."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = ""
    optimal_min: Optional[float] = None
    optimal_max: Optional[float] = None


class MedicalService:
    """Service for medical data interpretation and health insights."""
    
    def __init__(self):
        """Initialize medical service with reference ranges."""
        self.reference_ranges = self._load_reference_ranges()
    
    def _load_reference_ranges(self) -> Dict[str, ReferenceRange]:
        """
        Load reference ranges for common medical tests.
        
        Returns:
            Dictionary mapping test names to reference ranges
        """
        ranges = {
            # Blood Glucose
            'glucose': ReferenceRange(70, 100, 'mg/dL', 70, 90),
            'blood glucose': ReferenceRange(70, 100, 'mg/dL', 70, 90),
            'fasting glucose': ReferenceRange(70, 100, 'mg/dL', 70, 90),
            'random glucose': ReferenceRange(70, 140, 'mg/dL', 70, 120),
            
            # Cholesterol
            'total cholesterol': ReferenceRange(0, 200, 'mg/dL', 125, 180),
            'cholesterol': ReferenceRange(0, 200, 'mg/dL', 125, 180),
            'ldl': ReferenceRange(0, 100, 'mg/dL', 0, 70),
            'ldl cholesterol': ReferenceRange(0, 100, 'mg/dL', 0, 70),
            'hdl': ReferenceRange(40, 300, 'mg/dL', 50, 100),
            'hdl cholesterol': ReferenceRange(40, 300, 'mg/dL', 50, 100),
            'triglycerides': ReferenceRange(0, 150, 'mg/dL', 0, 100),
            
            # Complete Blood Count
            'hemoglobin': ReferenceRange(12.0, 17.5, 'g/dL', 13.5, 16.5),
            'hematocrit': ReferenceRange(35, 50, '%', 38, 46),
            'rbc': ReferenceRange(4.0, 5.5, 'M/μL', 4.2, 5.0),
            'wbc': ReferenceRange(4.0, 11.0, 'K/μL', 4.5, 10.0),
            'platelets': ReferenceRange(150, 400, 'K/μL', 150, 350),
            
            # Liver Function
            'alt': ReferenceRange(7, 56, 'U/L', 7, 40),
            'ast': ReferenceRange(10, 40, 'U/L', 10, 35),
            'bilirubin': ReferenceRange(0.1, 1.2, 'mg/dL', 0.1, 1.0),
            'albumin': ReferenceRange(3.5, 5.5, 'g/dL', 4.0, 5.0),
            
            # Kidney Function
            'creatinine': ReferenceRange(0.6, 1.3, 'mg/dL', 0.7, 1.2),
            'bun': ReferenceRange(7, 20, 'mg/dL', 7, 18),
            'urea': ReferenceRange(15, 43, 'mg/dL', 15, 40),
            
            # Thyroid
            'tsh': ReferenceRange(0.4, 4.0, 'mIU/L', 1.0, 2.5),
            't4': ReferenceRange(4.5, 12.0, 'μg/dL', 6.0, 10.0),
            't3': ReferenceRange(80, 200, 'ng/dL', 100, 180),
            
            # Vitamins & Minerals
            'vitamin d': ReferenceRange(30, 100, 'ng/mL', 40, 60),
            'vitamin b12': ReferenceRange(200, 900, 'pg/mL', 400, 800),
            'iron': ReferenceRange(60, 170, 'μg/dL', 70, 140),
            'calcium': ReferenceRange(8.5, 10.5, 'mg/dL', 9.0, 10.0),
        }
        
        return ranges
    
    def get_reference_range(self, test_name: str) -> Optional[ReferenceRange]:
        """
        Get reference range for a medical test.
        
        Args:
            test_name: Name of the medical test
            
        Returns:
            ReferenceRange object or None if not found
        """
        test_name_lower = test_name.lower().strip()
        return self.reference_ranges.get(test_name_lower)
    
    def assess_value(
        self,
        test_name: str,
        value: float,
        unit: str = ""
    ) -> Tuple[str, str, str]:
        """
        Assess a medical value against reference ranges.
        
        Args:
            test_name: Name of the medical test
            value: Test value
            unit: Unit of measurement
            
        Returns:
            Tuple of (status, severity, explanation)
        """
        ref_range = self.get_reference_range(test_name)
        
        if not ref_range:
            return "unknown", "info", "Reference range not available for this test"
        
        # Determine status
        if ref_range.min_value is not None and value < ref_range.min_value:
            status = "low"
            deviation = ((ref_range.min_value - value) / ref_range.min_value) * 100
            
            if deviation > 30:
                severity = "severe"
                explanation = f"Significantly below normal range (>{30:.0f}% below minimum)"
            elif deviation > 15:
                severity = "moderate"
                explanation = f"Moderately below normal range ({deviation:.0f}% below minimum)"
            else:
                severity = "mild"
                explanation = f"Slightly below normal range ({deviation:.0f}% below minimum)"
        
        elif ref_range.max_value is not None and value > ref_range.max_value:
            status = "high"
            deviation = ((value - ref_range.max_value) / ref_range.max_value) * 100
            
            if deviation > 30:
                severity = "severe"
                explanation = f"Significantly above normal range (>{30:.0f}% above maximum)"
            elif deviation > 15:
                severity = "moderate"
                explanation = f"Moderately above normal range ({deviation:.0f}% above maximum)"
            else:
                severity = "mild"
                explanation = f"Slightly above normal range ({deviation:.0f}% above maximum)"
        
        else:
            status = "normal"
            severity = "normal"
            explanation = "Within normal reference range"
        
        return status, severity, explanation
    
    def generate_insights(
        self,
        metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate health insights from medical metrics.
        
        Args:
            metrics: List of medical metrics with values
            
        Returns:
            List of health insights
        """
        insights = []
        
        # Count abnormal values
        abnormal_count = sum(
            1 for m in metrics
            if m.get('status') in ['low', 'high', 'critical']
        )
        
        # Summary insight
        if abnormal_count == 0:
            insights.append({
                'type': 'summary',
                'title': 'Overall Health Status: Good',
                'description': 'All measured values are within normal ranges. Continue maintaining your healthy lifestyle.',
                'severity': 'info',
                'priority': 100,
                'is_actionable': False
            })
        elif abnormal_count <= 2:
            insights.append({
                'type': 'summary',
                'title': 'Overall Health Status: Attention Needed',
                'description': f'{abnormal_count} value(s) outside normal range. Review the specific findings below and consult your healthcare provider.',
                'severity': 'warning',
                'priority': 90,
                'is_actionable': True
            })
        else:
            insights.append({
                'type': 'summary',
                'title': 'Overall Health Status: Requires Attention',
                'description': f'Multiple values ({abnormal_count}) are outside normal ranges. Please consult your healthcare provider for a comprehensive evaluation.',
                'severity': 'warning',
                'priority': 95,
                'is_actionable': True
            })
        
        # Specific insights for critical values
        for metric in metrics:
            if metric.get('severity') in ['moderate', 'severe']:
                insights.append(self._generate_metric_insight(metric))
        
        # Pattern-based insights
        insights.extend(self._generate_pattern_insights(metrics))
        
        return insights
    
    def _generate_metric_insight(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an insight for a specific abnormal metric."""
        test_name = metric.get('metric_name', '')
        status = metric.get('status', '')
        severity = metric.get('severity', '')
        
        insight = {
            'type': 'warning',
            'title': f'{test_name}: {status.title()}',
            'severity': severity,
            'priority': 80 if severity == 'severe' else 60,
            'is_actionable': True
        }
        
        # Generate specific recommendations
        test_lower = test_name.lower()
        
        if 'glucose' in test_lower and status == 'high':
            insight['description'] = (
                'Elevated blood glucose levels detected. This may indicate prediabetes or diabetes. '
                'Consider: reducing sugar intake, increasing physical activity, and consulting your doctor for further evaluation.'
            )
        elif 'cholesterol' in test_lower and status == 'high':
            insight['description'] = (
                'High cholesterol levels found. This increases cardiovascular risk. '
                'Recommendations: adopt a heart-healthy diet, increase exercise, reduce saturated fats, and consult your healthcare provider.'
            )
        elif 'hemoglobin' in test_lower and status == 'low':
            insight['description'] = (
                'Low hemoglobin levels indicate possible anemia. '
                'Consider: iron-rich foods, vitamin C for better iron absorption, and medical evaluation for underlying causes.'
            )
        else:
            insight['description'] = (
                f'Your {test_name} level is {status}. Please consult your healthcare provider to discuss '
                'this finding and determine appropriate next steps.'
            )
        
        return insight
    
    def _generate_pattern_insights(
        self,
        metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate insights based on patterns across multiple metrics."""
        insights = []
        
        # Check for metabolic syndrome indicators
        high_glucose = any(
            'glucose' in m.get('metric_name', '').lower() and m.get('status') == 'high'
            for m in metrics
        )
        high_cholesterol = any(
            'cholesterol' in m.get('metric_name', '').lower() and m.get('status') == 'high'
            for m in metrics
        )
        
        if high_glucose and high_cholesterol:
            insights.append({
                'type': 'recommendation',
                'title': 'Metabolic Health Attention',
                'description': (
                    'Both glucose and cholesterol levels are elevated, which may indicate metabolic syndrome risk. '
                    'A comprehensive lifestyle modification including diet, exercise, and stress management is recommended. '
                    'Please consult your healthcare provider for a complete metabolic assessment.'
                ),
                'severity': 'warning',
                'priority': 85,
                'is_actionable': True
            })
        
        return insights


# Global medical service instance
medical_service = MedicalService()
