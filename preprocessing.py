# data/preprocessing.py
"""
Data Preprocessing Pipeline for Healthcare Data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


class HealthcarePreprocessor:
    """Comprehensive data preprocessing for healthcare data."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = RobustScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_names = None
        self.categorical_cols = [
            'gender', 'ethnicity', 'smoking_status', 
            'alcohol_consumption', 'exercise_frequency'
        ]
        self.numerical_cols = None
        self.binary_cols = None
        self.target_cols = ['diabetes', 'heart_disease', 'kidney_disease', 'liver_disease']
        
    def identify_columns(self, df):
        """Identify column types."""
        exclude = ['patient_id'] + self.target_cols + self.categorical_cols
        self.numerical_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns 
            if col not in exclude
        ]
        self.binary_cols = [
            col for col in self.numerical_cols 
            if df[col].dropna().nunique() <= 2
        ]
        self.numerical_cols = [
            col for col in self.numerical_cols 
            if col not in self.binary_cols
        ]
    
    def handle_missing_values(self, df):
        """Handle missing values with appropriate strategies."""
        df_clean = df.copy()
        
        # Categorical: mode imputation
        for col in self.categorical_cols:
            if col in df_clean.columns:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Binary: mode imputation
        for col in self.binary_cols:
            if col in df_clean.columns:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Numerical: KNN imputation
        if self.numerical_cols:
            num_data = df_clean[self.numerical_cols].values
            num_imputed = self.imputer.fit_transform(num_data)
            df_clean[self.numerical_cols] = num_imputed
        
        return df_clean
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables."""
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    df_encoded[col] = self.label_encoders[col].transform(
                        df_encoded[col].astype(str)
                    )
        
        return df_encoded
    
    def engineer_features(self, df):
        """Create new meaningful features."""
        df_feat = df.copy()
        
        # Pulse Pressure
        if 'systolic_bp' in df_feat.columns and 'diastolic_bp' in df_feat.columns:
            df_feat['pulse_pressure'] = df_feat['systolic_bp'] - df_feat['diastolic_bp']
        
        # Mean Arterial Pressure
        if 'systolic_bp' in df_feat.columns and 'diastolic_bp' in df_feat.columns:
            df_feat['mean_arterial_pressure'] = (
                df_feat['diastolic_bp'] + 
                (df_feat['systolic_bp'] - df_feat['diastolic_bp']) / 3
            ).round(1)
        
        # Cholesterol Ratios
        if 'total_cholesterol' in df_feat.columns and 'hdl_cholesterol' in df_feat.columns:
            df_feat['cholesterol_ratio'] = (
                df_feat['total_cholesterol'] / df_feat['hdl_cholesterol'].clip(lower=1)
            ).round(2)
        
        if 'ldl_cholesterol' in df_feat.columns and 'hdl_cholesterol' in df_feat.columns:
            df_feat['ldl_hdl_ratio'] = (
                df_feat['ldl_cholesterol'] / df_feat['hdl_cholesterol'].clip(lower=1)
            ).round(2)
        
        # BMI Categories as numeric
        if 'bmi' in df_feat.columns:
            df_feat['bmi_category'] = pd.cut(
                df_feat['bmi'], 
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        
        # Age Groups
        if 'age' in df_feat.columns:
            df_feat['age_group'] = pd.cut(
                df_feat['age'],
                bins=[0, 30, 45, 60, 75, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        
        # Blood Pressure Category
        if 'systolic_bp' in df_feat.columns:
            df_feat['bp_category'] = pd.cut(
                df_feat['systolic_bp'],
                bins=[0, 120, 130, 140, 180, 300],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        
        # Symptom Count
        symptom_cols = [
            'chest_pain', 'shortness_of_breath', 'fatigue',
            'frequent_urination', 'blurred_vision', 'unexplained_weight_loss',
            'persistent_cough', 'dizziness', 'numbness_tingling', 'swelling'
        ]
        existing_symptoms = [col for col in symptom_cols if col in df_feat.columns]
        if existing_symptoms:
            df_feat['symptom_count'] = df_feat[existing_symptoms].sum(axis=1)
        
        # Risk factor count
        risk_cols = [
            'family_history_diabetes', 'family_history_heart_disease',
            'family_history_cancer', 'previous_hospitalization',
            'chronic_medication'
        ]
        existing_risks = [col for col in risk_cols if col in df_feat.columns]
        if existing_risks:
            df_feat['risk_factor_count'] = df_feat[existing_risks].sum(axis=1)
        
        return df_feat
    
    def scale_features(self, X, fit=True):
        """Scale numerical features."""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def preprocess(self, df, target='diabetes', fit=True):
        """Complete preprocessing pipeline."""
        print("📋 Starting preprocessing pipeline...")
        
        # Step 1: Identify columns
        self.identify_columns(df)
        print(f"  ✓ Identified {len(self.numerical_cols)} numerical, "
              f"{len(self.categorical_cols)} categorical, "
              f"{len(self.binary_cols)} binary columns")
        
        # Step 2: Handle missing values
        df_clean = self.handle_missing_values(df)
        missing_before = df.isnull().sum().sum()
        missing_after = df_clean.isnull().sum().sum()
        print(f"  ✓ Missing values: {missing_before} → {missing_after}")
        
        # Step 3: Feature Engineering
        df_feat = self.engineer_features(df_clean)
        print(f"  ✓ Feature engineering: {df.shape[1]} → {df_feat.shape[1]} columns")
        
        # Step 4: Encode categorical
        df_encoded = self.encode_categorical(df_feat, fit=fit)
        print(f"  ✓ Categorical encoding complete")
        
        # Step 5: Separate features and target
        drop_cols = ['patient_id'] + self.target_cols
        feature_cols = [col for col in df_encoded.columns if col not in drop_cols]
        
        X = df_encoded[feature_cols].fillna(0)
        y = df_encoded[target].astype(int)
        
        self.feature_names = feature_cols
        
        # Step 6: Scale features
        X_scaled = self.scale_features(X.values, fit=fit)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        print(f"  ✓ Final shape: X={X_scaled.shape}, y={y.shape}")
        print(f"  ✓ Target distribution: {y.value_counts().to_dict()}")
        
        return X_scaled, y, feature_cols
    
    def balance_dataset(self, X, y, strategy='smote'):
        """Balance the dataset using SMOTE or combined sampling."""
        print(f"\n⚖️ Balancing dataset (strategy: {strategy})...")
        print(f"  Before: {dict(pd.Series(y).value_counts())}")
        
        if strategy == 'smote':
            sampler = SMOTE(random_state=42)
        elif strategy == 'combined':
            over = SMOTE(sampling_strategy=0.8, random_state=42)
            under = RandomUnderSampler(sampling_strategy=0.9, random_state=42)
            sampler = ImbPipeline(steps=[('over', over), ('under', under)])
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        print(f"  After: {dict(pd.Series(y_balanced).value_counts())}")
        
        return X_balanced, y_balanced
