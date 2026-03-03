"""
Medical Training Data Generator
Creates realistic synthetic medical lab data for ML model training
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import json

class MedicalDataGenerator:
    """Generate synthetic but realistic medical lab data"""
    
    def __init__(self):
        # Normal ranges for common lab tests (based on standard medical references)
        self.lab_parameters = {
            'Hemoglobin': {'unit': 'g/dL', 'normal_min': 12.0, 'normal_max': 16.0, 'mean': 14.0, 'std': 1.5},
            'WBC Count': {'unit': 'cells/μL', 'normal_min': 4000, 'normal_max': 11000, 'mean': 7500, 'std': 2000},
            'RBC Count': {'unit': 'million/μL', 'normal_min': 4.5, 'normal_max': 5.5, 'mean': 5.0, 'std': 0.4},
            'Platelet Count': {'unit': 'cells/μL', 'normal_min': 150000, 'normal_max': 400000, 'mean': 275000, 'std': 70000},
            'Blood Glucose': {'unit': 'mg/dL', 'normal_min': 70, 'normal_max': 100, 'mean': 85, 'std': 12},
            'HbA1c': {'unit': '%', 'normal_min': 4.0, 'normal_max': 5.6, 'mean': 4.8, 'std': 0.5},
            'Total Cholesterol': {'unit': 'mg/dL', 'normal_min': 125, 'normal_max': 200, 'mean': 162, 'std': 30},
            'LDL Cholesterol': {'unit': 'mg/dL', 'normal_min': 0, 'normal_max': 100, 'mean': 90, 'std': 25},
            'HDL Cholesterol': {'unit': 'mg/dL', 'normal_min': 40, 'normal_max': 60, 'mean': 50, 'std': 10},
            'Triglycerides': {'unit': 'mg/dL', 'normal_min': 0, 'normal_max': 150, 'mean': 100, 'std': 40},
            'Creatinine': {'unit': 'mg/dL', 'normal_min': 0.6, 'normal_max': 1.2, 'mean': 0.9, 'std': 0.2},
            'Blood Urea': {'unit': 'mg/dL', 'normal_min': 7, 'normal_max': 20, 'mean': 13, 'std': 4},
            'SGPT (ALT)': {'unit': 'U/L', 'normal_min': 7, 'normal_max': 56, 'mean': 30, 'std': 15},
            'SGOT (AST)': {'unit': 'U/L', 'normal_min': 10, 'normal_max': 40, 'mean': 25, 'std': 10},
            'Bilirubin Total': {'unit': 'mg/dL', 'normal_min': 0.1, 'normal_max': 1.2, 'mean': 0.6, 'std': 0.3},
            'Albumin': {'unit': 'g/dL', 'normal_min': 3.5, 'normal_max': 5.5, 'mean': 4.5, 'std': 0.6},
            'TSH': {'unit': 'mIU/L', 'normal_min': 0.4, 'normal_max': 4.0, 'mean': 2.0, 'std': 1.0},
            'Vitamin D': {'unit': 'ng/mL', 'normal_min': 30, 'normal_max': 100, 'mean': 50, 'std': 20},
            'Vitamin B12': {'unit': 'pg/mL', 'normal_min': 200, 'normal_max': 900, 'mean': 500, 'std': 150},
            'Calcium': {'unit': 'mg/dL', 'normal_min': 8.5, 'normal_max': 10.5, 'mean': 9.5, 'std': 0.6},
        }
    
    def generate_normal_samples(self, n_samples: int = 700) -> pd.DataFrame:
        """Generate normal (healthy) lab samples"""
        data = []
        for _ in range(n_samples):
            sample = {}
            for param, config in self.lab_parameters.items():
                # Generate value within normal range with slight variations
                value = np.random.normal(config['mean'], config['std'] * 0.5)
                value = np.clip(value, config['normal_min'], config['normal_max'])
                sample[param] = round(value, 2)
            sample['anomaly'] = 0
            sample['condition'] = 'healthy'
            data.append(sample)
        return pd.DataFrame(data)
    
    def generate_anomalous_samples(self, n_samples: int = 300) -> pd.DataFrame:
        """Generate anomalous (unhealthy) lab samples"""
        data = []
        conditions = [
            'diabetes', 'anemia', 'liver_disease', 'kidney_disease',
            'thyroid_disorder', 'high_cholesterol', 'vitamin_deficiency'
        ]
        
        for _ in range(n_samples):
            sample = {}
            condition = np.random.choice(conditions)
            
            for param, config in self.lab_parameters.items():
                # Generate anomalous values based on condition
                if self._is_affected_by_condition(param, condition):
                    # Generate out-of-range value
                    if np.random.random() < 0.5:
                        # Too low
                        value = np.random.normal(
                            config['normal_min'] - config['std'] * 1.5,
                            config['std'] * 0.5
                        )
                    else:
                        # Too high
                        value = np.random.normal(
                            config['normal_max'] + config['std'] * 1.5,
                            config['std'] * 0.5
                        )
                else:
                    # Normal value
                    value = np.random.normal(config['mean'], config['std'] * 0.5)
                    value = np.clip(value, config['normal_min'], config['normal_max'])
                
                sample[param] = round(value, 2)
            
            sample['anomaly'] = 1
            sample['condition'] = condition
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def _is_affected_by_condition(self, param: str, condition: str) -> bool:
        """Determine if a parameter is affected by a medical condition"""
        condition_effects = {
            'diabetes': ['Blood Glucose', 'HbA1c', 'Triglycerides'],
            'anemia': ['Hemoglobin', 'RBC Count', 'Vitamin B12'],
            'liver_disease': ['SGPT (ALT)', 'SGOT (AST)', 'Bilirubin Total', 'Albumin'],
            'kidney_disease': ['Creatinine', 'Blood Urea', 'Calcium'],
            'thyroid_disorder': ['TSH', 'Cholesterol'],
            'high_cholesterol': ['Total Cholesterol', 'LDL Cholesterol', 'Triglycerides'],
            'vitamin_deficiency': ['Vitamin D', 'Vitamin B12', 'Calcium']
        }
        return param in condition_effects.get(condition, [])
    
    def generate_complete_dataset(self, save_path: str = None) -> pd.DataFrame:
        """Generate complete training dataset with 1000 samples"""
        normal_data = self.generate_normal_samples(700)
        anomalous_data = self.generate_anomalous_samples(300)
        
        dataset = pd.concat([normal_data, anomalous_data], ignore_index=True)
        dataset = dataset.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        if save_path:
            dataset.to_csv(save_path, index=False)
            print(f"Dataset saved to {save_path}")
        
        return dataset


# Comprehensive Medication Database
MEDICATION_DATABASE = {
    # Antibiotics
    'amoxicillin': {'class': 'Antibiotic', 'type': 'Penicillin', 'common_doses': ['250mg', '500mg']},
    'azithromycin': {'class': 'Antibiotic', 'type': 'Macrolide', 'common_doses': ['250mg', '500mg']},
    'ciprofloxacin': {'class': 'Antibiotic', 'type': 'Fluoroquinolone', 'common_doses': ['250mg', '500mg', '750mg']},
    'doxycycline': {'class': 'Antibiotic', 'type': 'Tetracycline', 'common_doses': ['100mg']},
    'metronidazole': {'class': 'Antibiotic', 'type': 'Nitroimidazole', 'common_doses': ['400mg', '500mg']},
    'cephalexin': {'class': 'Antibiotic', 'type': 'Cephalosporin', 'common_doses': ['250mg', '500mg']},
    
    # Diabetes
    'metformin': {'class': 'Antidiabetic', 'type': 'Biguanide', 'common_doses': ['500mg', '850mg', '1000mg']},
    'glipizide': {'class': 'Antidiabetic', 'type': 'Sulfonylurea', 'common_doses': ['5mg', '10mg']},
    'insulin': {'class': 'Antidiabetic', 'type': 'Hormone', 'common_doses': ['varies by type']},
    'sitagliptin': {'class': 'Antidiabetic', 'type': 'DPP-4 Inhibitor', 'common_doses': ['25mg', '50mg', '100mg']},
    
    # Cardiovascular
    'atorvastatin': {'class': 'Statin', 'type': 'Cholesterol-lowering', 'common_doses': ['10mg', '20mg', '40mg', '80mg']},
    'rosuvastatin': {'class': 'Statin', 'type': 'Cholesterol-lowering', 'common_doses': ['5mg', '10mg', '20mg', '40mg']},
    'amlodipine': {'class': 'Antihypertensive', 'type': 'Calcium Channel Blocker', 'common_doses': ['2.5mg', '5mg', '10mg']},
    'lisinopril': {'class': 'Antihypertensive', 'type': 'ACE Inhibitor', 'common_doses': ['2.5mg', '5mg', '10mg', '20mg']},
    'losartan': {'class': 'Antihypertensive', 'type': 'ARB', 'common_doses': ['25mg', '50mg', '100mg']},
    'metoprolol': {'class': 'Antihypertensive', 'type': 'Beta Blocker', 'common_doses': ['25mg', '50mg', '100mg']},
    'aspirin': {'class': 'Antiplatelet', 'type': 'NSAID', 'common_doses': ['75mg', '81mg', '325mg']},
    'clopidogrel': {'class': 'Antiplatelet', 'type': 'P2Y12 Inhibitor', 'common_doses': ['75mg']},
    'warfarin': {'class': 'Anticoagulant', 'type': 'Vitamin K Antagonist', 'common_doses': ['varies']},
    
    # Pain/Inflammation
    'ibuprofen': {'class': 'NSAID', 'type': 'Anti-inflammatory', 'common_doses': ['200mg', '400mg', '600mg', '800mg']},
    'naproxen': {'class': 'NSAID', 'type': 'Anti-inflammatory', 'common_doses': ['250mg', '500mg']},
    'diclofenac': {'class': 'NSAID', 'type': 'Anti-inflammatory', 'common_doses': ['50mg', '75mg']},
    'acetaminophen': {'class': 'Analgesic', 'type': 'Pain reliever', 'common_doses': ['325mg', '500mg', '650mg']},
    'tramadol': {'class': 'Opioid', 'type': 'Pain reliever', 'common_doses': ['50mg', '100mg']},
    
    # Gastrointestinal
    'omeprazole': {'class': 'PPI', 'type': 'Acid reducer', 'common_doses': ['20mg', '40mg']},
    'pantoprazole': {'class': 'PPI', 'type': 'Acid reducer', 'common_doses': ['20mg', '40mg']},
    'ranitidine': {'class': 'H2 Blocker', 'type': 'Acid reducer', 'common_doses': ['150mg', '300mg']},
    'ondansetron': {'class': 'Antiemetic', 'type': 'Anti-nausea', 'common_doses': ['4mg', '8mg']},
    
    # Respiratory
    'salbutamol': {'class': 'Bronchodilator', 'type': 'Beta-2 Agonist', 'common_doses': ['inhaler']},
    'montelukast': {'class': 'Antiasthmatic', 'type': 'Leukotriene Inhibitor', 'common_doses': ['4mg', '5mg', '10mg']},
    'cetirizine': {'class': 'Antihistamine', 'type': 'Allergy', 'common_doses': ['5mg', '10mg']},
    'loratadine': {'class': 'Antihistamine', 'type': 'Allergy', 'common_doses': ['10mg']},
    
    # Thyroid
    'levothyroxine': {'class': 'Thyroid Hormone', 'type': 'Replacement', 'common_doses': ['25mcg', '50mcg', '75mcg', '100mcg']},
    
    # Mental Health
    'sertraline': {'class': 'Antidepressant', 'type': 'SSRI', 'common_doses': ['25mg', '50mg', '100mg']},
    'escitalopram': {'class': 'Antidepressant', 'type': 'SSRI', 'common_doses': ['5mg', '10mg', '20mg']},
    'fluoxetine': {'class': 'Antidepressant', 'type': 'SSRI', 'common_doses': ['10mg', '20mg', '40mg']},
    'alprazolam': {'class': 'Anxiolytic', 'type': 'Benzodiazepine', 'common_doses': ['0.25mg', '0.5mg', '1mg']},
    'lorazepam': {'class': 'Anxiolytic', 'type': 'Benzodiazepine', 'common_doses': ['0.5mg', '1mg', '2mg']},
    
    # Vitamins/Supplements
    'vitamin d': {'class': 'Vitamin', 'type': 'Supplement', 'common_doses': ['1000IU', '2000IU', '5000IU']},
    'vitamin b12': {'class': 'Vitamin', 'type': 'Supplement', 'common_doses': ['500mcg', '1000mcg']},
    'folic acid': {'class': 'Vitamin', 'type': 'Supplement', 'common_doses': ['400mcg', '1mg', '5mg']},
    'calcium': {'class': 'Mineral', 'type': 'Supplement', 'common_doses': ['500mg', '600mg', '1000mg']},
    'iron': {'class': 'Mineral', 'type': 'Supplement', 'common_doses': ['ferrous sulfate 325mg']},
}


def get_medication_info(med_name: str) -> Dict[str, Any]:
    """Get detailed information about a medication"""
    med_lower = med_name.lower().strip()
    
    # Direct lookup
    if med_lower in MEDICATION_DATABASE:
        return {
            'name': med_name,
            'found': True,
            **MEDICATION_DATABASE[med_lower]
        }
    
    # Fuzzy matching
    for db_med, info in MEDICATION_DATABASE.items():
        if db_med in med_lower or med_lower in db_med:
            return {
                'name': med_name,
                'found': True,
                'matched_as': db_med,
                **info
            }
    
    return {
        'name': med_name,
        'found': False,
        'class': 'Unknown',
        'type': 'Unknown'
    }
