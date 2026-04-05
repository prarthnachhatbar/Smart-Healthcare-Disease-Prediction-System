import numpy as np
import pandas as pd
import random


class HealthcareDataGenerator:
    def __init__(self, n_samples=5000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

    def generate_patient_demographics(self):
        n = self.n_samples
        ages = np.random.normal(50, 18, n).clip(18, 95).astype(int)
        genders = np.random.choice(
            ['Male', 'Female'], n, p=[0.48, 0.52]
        )
        ethnicities = np.random.choice(
            ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'],
            n, p=[0.40, 0.20, 0.20, 0.12, 0.08]
        )
        bmi = np.random.normal(27.5, 6.0, n).clip(15, 55).round(1)
        smoking_status = np.random.choice(
            ['Never', 'Former', 'Current'], n, p=[0.55, 0.25, 0.20]
        )
        alcohol_consumption = np.random.choice(
            ['None', 'Light', 'Moderate', 'Heavy'],
            n, p=[0.30, 0.35, 0.25, 0.10]
        )
        exercise_frequency = np.random.choice(
            ['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active'],
            n, p=[0.20, 0.25, 0.30, 0.15, 0.10]
        )
        return pd.DataFrame({
            'patient_id': [
                f'P{str(i).zfill(5)}' for i in range(n)
            ],
            'age': ages,
            'gender': genders,
            'ethnicity': ethnicities,
            'bmi': bmi,
            'smoking_status': smoking_status,
            'alcohol_consumption': alcohol_consumption,
            'exercise_frequency': exercise_frequency
        })

    def generate_vital_signs(self, demographics):
        n = len(demographics)
        ages = demographics['age'].values
        bmis = demographics['bmi'].values
        systolic_bp = (
            100 + 0.4*ages + 0.5*bmis
            + np.random.normal(0, 10, n)
        ).clip(85, 200).astype(int)
        diastolic_bp = (
            60 + 0.2*ages + 0.3*bmis
            + np.random.normal(0, 8, n)
        ).clip(55, 130).astype(int)
        heart_rate = (
            72 + 0.05*ages - 0.1*bmis
            + np.random.normal(0, 10, n)
        ).clip(50, 120).astype(int)
        respiratory_rate = (
            16 + 0.02*ages
            + np.random.normal(0, 2, n)
        ).clip(12, 30).astype(int)
        temperature = np.random.normal(
            98.6, 0.5, n
        ).clip(96.0, 103.0).round(1)
        oxygen_saturation = (
            98 - 0.02*ages
            + np.random.normal(0, 1.5, n)
        ).clip(88, 100).round(1)
        return pd.DataFrame({
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'respiratory_rate': respiratory_rate,
            'temperature': temperature,
            'oxygen_saturation': oxygen_saturation
        })

    def generate_lab_results(self, demographics):
        n = len(demographics)
        ages = demographics['age'].values
        bmis = demographics['bmi'].values
        blood_glucose = (
            85 + 0.3*ages + 0.8*bmis
            + np.random.normal(0, 20, n)
        ).clip(60, 400).round(1)
        hba1c = (
            4.5 + 0.01*ages + 0.03*bmis
            + np.random.normal(0, 0.8, n)
        ).clip(3.5, 14.0).round(1)
        total_cholesterol = (
            170 + 0.5*ages + 0.8*bmis
            + np.random.normal(0, 25, n)
        ).clip(100, 350).round(1)
        hdl_cholesterol = (
            55 - 0.1*bmis
            + np.random.normal(0, 12, n)
        ).clip(20, 100).round(1)
        ldl_cholesterol = (
            100 + 0.4*ages + 0.6*bmis
            + np.random.normal(0, 20, n)
        ).clip(40, 250).round(1)
        triglycerides = (
            120 + 0.5*bmis + 0.3*ages
            + np.random.normal(0, 40, n)
        ).clip(40, 500).round(1)
        creatinine = (
            0.8 + 0.005*ages
            + np.random.normal(0, 0.2, n)
        ).clip(0.4, 5.0).round(2)
        hemoglobin = (
            14.0 - 0.02*ages
            + np.random.normal(0, 1.2, n)
        ).clip(7.0, 18.0).round(1)
        wbc_count = (
            7.0 + np.random.normal(0, 2.0, n)
        ).clip(2.0, 20.0).round(1)
        platelet_count = (
            250 + np.random.normal(0, 50, n)
        ).clip(100, 500).astype(int)
        return pd.DataFrame({
            'blood_glucose': blood_glucose,
            'hba1c': hba1c,
            'total_cholesterol': total_cholesterol,
            'hdl_cholesterol': hdl_cholesterol,
            'ldl_cholesterol': ldl_cholesterol,
            'triglycerides': triglycerides,
            'creatinine': creatinine,
            'hemoglobin': hemoglobin,
            'wbc_count': wbc_count,
            'platelet_count': platelet_count
        })

    def generate_medical_history(self):
        n = self.n_samples
        return pd.DataFrame({
            'family_history_diabetes': np.random.choice(
                [0, 1], n, p=[0.7, 0.3]
            ),
            'family_history_heart_disease': np.random.choice(
                [0, 1], n, p=[0.65, 0.35]
            ),
            'family_history_cancer': np.random.choice(
                [0, 1], n, p=[0.75, 0.25]
            ),
            'previous_hospitalization': np.random.choice(
                [0, 1], n, p=[0.6, 0.4]
            ),
            'chronic_medication': np.random.choice(
                [0, 1], n, p=[0.55, 0.45]
            ),
            'allergies': np.random.choice(
                [0, 1], n, p=[0.7, 0.3]
            ),
            'previous_surgery': np.random.choice(
                [0, 1], n, p=[0.65, 0.35]
            )
        })

    def generate_symptoms(self):
        n = self.n_samples
        return pd.DataFrame({
            'chest_pain': np.random.choice(
                [0, 1], n, p=[0.75, 0.25]
            ),
            'shortness_of_breath': np.random.choice(
                [0, 1], n, p=[0.70, 0.30]
            ),
            'fatigue': np.random.choice(
                [0, 1], n, p=[0.55, 0.45]
            ),
            'frequent_urination': np.random.choice(
                [0, 1], n, p=[0.72, 0.28]
            ),
            'blurred_vision': np.random.choice(
                [0, 1], n, p=[0.80, 0.20]
            ),
            'unexplained_weight_loss': np.random.choice(
                [0, 1], n, p=[0.82, 0.18]
            ),
            'persistent_cough': np.random.choice(
                [0, 1], n, p=[0.78, 0.22]
            ),
            'dizziness': np.random.choice(
                [0, 1], n, p=[0.75, 0.25]
            ),
            'numbness_tingling': np.random.choice(
                [0, 1], n, p=[0.80, 0.20]
            ),
            'swelling': np.random.choice(
                [0, 1], n, p=[0.78, 0.22]
            )
        })

    def generate_disease_labels(self, full_data):
        n = len(full_data)
        diabetes_score = (
            0.03 * full_data['age']
            + 0.05 * full_data['bmi']
            + 0.15 * full_data['blood_glucose'] / 100
            + 0.3 * full_data['hba1c'] / 6
            + 0.2 * full_data['family_history_diabetes']
            + 0.15 * full_data['frequent_urination']
            + 0.1 * full_data['fatigue']
            + 0.1 * (
                full_data['smoking_status'] == 'Current'
            ).astype(int)
            + np.random.normal(0, 0.3, n)
        )
        diabetes = (
            diabetes_score > np.percentile(diabetes_score, 75)
        ).astype(int)

        heart_score = (
            0.04 * full_data['age']
            + 0.03 * full_data['bmi']
            + 0.02 * full_data['systolic_bp']
            + 0.15 * full_data['total_cholesterol'] / 200
            + 0.2 * full_data['chest_pain']
            + 0.15 * full_data['shortness_of_breath']
            + 0.15 * full_data['family_history_heart_disease']
            + 0.15 * (
                full_data['smoking_status'] == 'Current'
            ).astype(int)
            + 0.1 * (
                full_data['exercise_frequency'] == 'Sedentary'
            ).astype(int)
            + np.random.normal(0, 0.3, n)
        )
        heart_disease = (
            heart_score > np.percentile(heart_score, 72)
        ).astype(int)

        kidney_score = (
            0.03 * full_data['age']
            + 0.4 * full_data['creatinine']
            + 0.02 * full_data['systolic_bp']
            + 0.15 * diabetes
            + 0.1 * full_data['swelling']
            + 0.1 * full_data['fatigue']
            + np.random.normal(0, 0.3, n)
        )
        kidney_disease = (
            kidney_score > np.percentile(kidney_score, 80)
        ).astype(int)

        liver_score = (
            0.02 * full_data['age']
            + 0.04 * full_data['bmi']
            + 0.25 * (
                full_data['alcohol_consumption'] == 'Heavy'
            ).astype(int)
            + 0.15 * full_data['fatigue']
            + 0.1 * full_data['unexplained_weight_loss']
            + np.random.normal(0, 0.3, n)
        )
        liver_disease = (
            liver_score > np.percentile(liver_score, 82)
        ).astype(int)

        return pd.DataFrame({
            'diabetes': diabetes,
            'heart_disease': heart_disease,
            'kidney_disease': kidney_disease,
            'liver_disease': liver_disease
        })

    def generate_complete_dataset(self):
        demographics = self.generate_patient_demographics()
        vitals = self.generate_vital_signs(demographics)
        labs = self.generate_lab_results(demographics)
        history = self.generate_medical_history()
        symptoms = self.generate_symptoms()
        full_data = pd.concat(
            [demographics, vitals, labs, history, symptoms],
            axis=1
        )
        labels = self.generate_disease_labels(full_data)
        dataset = pd.concat([full_data, labels], axis=1)
        mask = np.random.random(dataset.shape) < 0.02
        protected = [
            'patient_id', 'diabetes', 'heart_disease',
            'kidney_disease', 'liver_disease'
        ]
        for col in protected:
            mask[:, dataset.columns.get_loc(col)] = False
        original = pd.concat([full_data, labels], axis=1)
        dataset = dataset.mask(mask)
        for col in protected:
            dataset[col] = original[col]
        return dataset