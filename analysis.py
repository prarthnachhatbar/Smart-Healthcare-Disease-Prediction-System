# eda/analysis.py
"""
Comprehensive Exploratory Data Analysis for Healthcare Data
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


class HealthcareEDA:
    """Comprehensive EDA toolkit for healthcare data."""
    
    def __init__(self, df, target_cols=None):
        self.df = df.copy()
        self.target_cols = target_cols or [
            'diabetes', 'heart_disease', 'kidney_disease', 'liver_disease'
        ]
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#2ECC71',
            'danger': '#E74C3C',
            'warning': '#F39C12',
            'info': '#3498DB',
            'palette': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
                       '#2ECC71', '#9B59B6', '#1ABC9C', '#E74C3C', '#F39C12']
        }
    
    def dataset_overview(self):
        """Generate dataset overview statistics."""
        overview = {
            'Total Patients': len(self.df),
            'Total Features': len(self.df.columns),
            'Numerical Features': len(self.numerical_cols),
            'Categorical Features': len(self.categorical_cols),
            'Missing Values (%)': round(
                self.df.isnull().sum().sum() / 
                (self.df.shape[0] * self.df.shape[1]) * 100, 2
            ),
            'Duplicate Rows': self.df.duplicated().sum()
        }
        
        # Disease prevalence
        for target in self.target_cols:
            if target in self.df.columns:
                prevalence = self.df[target].mean() * 100
                overview[f'{target.replace("_", " ").title()} Prevalence (%)'] = round(prevalence, 1)
        
        return overview
    
    def plot_disease_distribution(self):
        """Plot disease distribution across all target variables."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[t.replace('_', ' ').title() for t in self.target_cols],
            specs=[[{'type': 'pie'}, {'type': 'pie'}],
                   [{'type': 'pie'}, {'type': 'pie'}]]
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        colors_pair = [self.colors['success'], self.colors['danger']]
        
        for idx, target in enumerate(self.target_cols):
            if target in self.df.columns:
                counts = self.df[target].value_counts()
                row, col = positions[idx]
                fig.add_trace(
                    go.Pie(
                        labels=['Negative', 'Positive'],
                        values=[counts.get(0, 0), counts.get(1, 0)],
                        marker_colors=colors_pair,
                        hole=0.4,
                        textinfo='label+percent',
                        name=target
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="🏥 Disease Distribution Overview",
            title_font_size=20,
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        return fig
    
    def plot_age_distribution_by_disease(self, target='diabetes'):
        """Plot age distribution colored by disease status."""
        if target not in self.df.columns or 'age' not in self.df.columns:
            return None
        
        df_plot = self.df.dropna(subset=['age', target])
        
        fig = go.Figure()
        
        for status, color, name in [(0, self.colors['success'], 'Healthy'), 
                                     (1, self.colors['danger'], 'Diagnosed')]:
            data = df_plot[df_plot[target] == status]['age']
            fig.add_trace(go.Histogram(
                x=data,
                name=name,
                marker_color=color,
                opacity=0.7,
                nbinsx=30
            ))
        
        fig.update_layout(
            title=f"📊 Age Distribution by {target.replace('_', ' ').title()} Status",
            xaxis_title="Age",
            yaxis_title="Count",
            barmode='overlay',
            template='plotly_white',
            height=450,
            legend=dict(x=0.02, y=0.98)
        )
        return fig
    
    def plot_correlation_heatmap(self, max_features=20):
        """Plot interactive correlation heatmap."""
        num_cols = [col for col in self.numerical_cols 
                   if col not in ['patient_id']][:max_features]
        
        corr_matrix = self.df[num_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🔗 Feature Correlation Heatmap",
            height=700,
            width=800,
            template='plotly_white'
        )
        return fig
    
    def plot_feature_distributions(self, features=None, ncols=3):
        """Plot distribution of numerical features."""
        if features is None:
            features = [col for col in self.numerical_cols 
                       if col not in self.target_cols + ['patient_id']][:12]
        
        nrows = (len(features) + ncols - 1) // ncols
        
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[f.replace('_', ' ').title() for f in features],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        for idx, feature in enumerate(features):
            row = idx // ncols + 1
            col = idx % ncols + 1
            
            data = self.df[feature].dropna()
            
            fig.add_trace(
                go.Histogram(
                    x=data,
                    marker_color=self.colors['palette'][idx % len(self.colors['palette'])],
                    opacity=0.8,
                    nbinsx=30,
                    name=feature,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="📈 Feature Distributions",
            height=300 * nrows,
            template='plotly_white',
            showlegend=False
        )
        return fig
    
    def plot_vital_signs_radar(self, target='diabetes'):
        """Radar chart comparing vital signs between healthy and diagnosed."""
        vital_features = [
            'systolic_bp', 'diastolic_bp', 'heart_rate', 
            'respiratory_rate', 'temperature', 'oxygen_saturation'
        ]
        
        existing = [f for f in vital_features if f in self.df.columns]
        if not existing or target not in self.df.columns:
            return None
        
        healthy = self.df[self.df[target] == 0][existing].mean()
        diagnosed = self.df[self.df[target] == 1][existing].mean()
        
        # Normalize to 0-1 range for radar
        all_vals = pd.concat([healthy, diagnosed], axis=1)
        normalized = (all_vals - all_vals.min(axis=1).values.reshape(-1,1)) / \
                     (all_vals.max(axis=1).values.reshape(-1,1) - 
                      all_vals.min(axis=1).values.reshape(-1,1) + 1e-10)
        
        categories = [f.replace('_', ' ').title() for f in existing]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized.iloc[:, 0].tolist() + [normalized.iloc[0, 0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Healthy',
            line_color=self.colors['success'],
            opacity=0.6
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=normalized.iloc[:, 1].tolist() + [normalized.iloc[0, 1]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Diagnosed',
            line_color=self.colors['danger'],
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"🎯 Vital Signs Comparison - {target.replace('_', ' ').title()}",
            height=500,
            template='plotly_white'
        )
        return fig
    
    def plot_risk_factor_analysis(self, target='diabetes'):
        """Analyze risk factors for a specific disease."""
        risk_features = [
            'age', 'bmi', 'systolic_bp', 'blood_glucose', 'hba1c',
            'total_cholesterol', 'ldl_cholesterol', 'triglycerides',
            'creatinine', 'hemoglobin'
        ]
        
        existing = [f for f in risk_features if f in self.df.columns]
        
        fig = make_subplots(
            rows=2, cols=5,
            subplot_titles=[f.replace('_', ' ').title() for f in existing[:10]],
            vertical_spacing=0.15
        )
        
        for idx, feature in enumerate(existing[:10]):
            row = idx // 5 + 1
            col = idx % 5 + 1
            
            for status, color, name in [(0, self.colors['success'], 'Healthy'),
                                         (1, self.colors['danger'], 'Diagnosed')]:
                data = self.df[self.df[target] == status][feature].dropna()
                fig.add_trace(
                    go.Box(
                        y=data,
                        name=name,
                        marker_color=color,
                        showlegend=(idx == 0),
                        legendgroup=name
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=f"⚠️ Risk Factor Analysis - {target.replace('_', ' ').title()}",
            height=500,
            template='plotly_white',
            boxmode='group'
        )
        return fig
    
    def plot_demographic_analysis(self):
        """Analyze demographic distribution."""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Gender Distribution', 'Ethnicity Distribution',
                'Smoking Status', 'BMI Distribution',
                'Age Distribution', 'Exercise Frequency'
            ],
            specs=[
                [{'type': 'pie'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'histogram'}, {'type': 'bar'}]
            ]
        )
        
        # Gender Pie
        if 'gender' in self.df.columns:
            gender_counts = self.df['gender'].value_counts()
            fig.add_trace(
                go.Pie(labels=gender_counts.index, values=gender_counts.values,
                       hole=0.4, marker_colors=[self.colors['primary'], self.colors['secondary']]),
                row=1, col=1
            )
        
        # Ethnicity Bar
        if 'ethnicity' in self.df.columns:
            eth_counts = self.df['ethnicity'].value_counts()
            fig.add_trace(
                go.Bar(x=eth_counts.index, y=eth_counts.values,
                       marker_color=self.colors['palette'][:len(eth_counts)],
                       showlegend=False),
                row=1, col=2
            )
        
        # Smoking Bar
        if 'smoking_status' in self.df.columns:
            smoke_counts = self.df['smoking_status'].value_counts()
            fig.add_trace(
                go.Bar(x=smoke_counts.index, y=smoke_counts.values,
                       marker_color=[self.colors['success'], self.colors['warning'], self.colors['danger']],
                       showlegend=False),
                row=1, col=3
            )
        
        # BMI Histogram
        if 'bmi' in self.df.columns:
            fig.add_trace(
                go.Histogram(x=self.df['bmi'].dropna(), nbinsx=30,
                           marker_color=self.colors['info'], showlegend=False),
                row=2, col=1
            )
        
        # Age Histogram
        if 'age' in self.df.columns:
            fig.add_trace(
                go.Histogram(x=self.df['age'].dropna(), nbinsx=30,
                           marker_color=self.colors['primary'], showlegend=False),
                row=2, col=2
            )
        
        # Exercise Bar
        if 'exercise_frequency' in self.df.columns:
            ex_counts = self.df['exercise_frequency'].value_counts()
            fig.add_trace(
                go.Bar(x=ex_counts.index, y=ex_counts.values,
                       marker_color=self.colors['palette'][:len(ex_counts)],
                       showlegend=False),
                row=2, col=3
            )
        
        fig.update_layout(
            title="👥 Patient Demographics Overview",
            height=700,
            template='plotly_white',
            showlegend=False
        )
        return fig
    
    def plot_missing_values(self):
        """Visualize missing value patterns."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)
        
        if len(missing) == 0:
            return None
        
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        fig = go.Figure()
        
        colors = ['#2ECC71' if pct < 5 else '#F39C12' if pct < 15 else '#E74C3C' 
                  for pct in missing_pct]
        
        fig.add_trace(go.Bar(
            y=missing_pct.index,
            x=missing_pct.values,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.1f}%' for v in missing_pct.values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="🔍 Missing Values Analysis",
            xaxis_title="Missing Percentage (%)",
            yaxis_title="Feature",
            height=max(400, len(missing) * 25),
            template='plotly_white'
        )
        return fig
    
    def plot_pairwise_scatter(self, features=None, target='diabetes'):
        """Interactive pairwise scatter plots."""
        if features is None:
            features = ['age', 'bmi', 'blood_glucose', 'systolic_bp']
        
        existing = [f for f in features if f in self.df.columns]
        
        if target in self.df.columns:
            df_plot = self.df[existing + [target]].dropna()
            df_plot[target] = df_plot[target].map({0: 'Healthy', 1: 'Diagnosed'})
            
            fig = px.scatter_matrix(
                df_plot,
                dimensions=existing,
                color=target,
                color_discrete_map={'Healthy': self.colors['success'], 
                                   'Diagnosed': self.colors['danger']},
                opacity=0.5,
                title=f"🔄 Pairwise Feature Relationships - {target.replace('_', ' ').title()}"
            )
        else:
            df_plot = self.df[existing].dropna()
            fig = px.scatter_matrix(df_plot, dimensions=existing, opacity=0.5)
        
        fig.update_layout(height=700, template='plotly_white')
        fig.update_traces(diagonal_visible=False, marker=dict(size=3))
        return fig
    
    def statistical_summary(self, target='diabetes'):
        """Generate statistical summary with hypothesis tests."""
        results = []
        
        num_features = [col for col in self.numerical_cols 
                       if col not in self.target_cols + ['patient_id']]
        
        for feature in num_features:
            if feature in self.df.columns and target in self.df.columns:
                group_0 = self.df[self.df[target] == 0][feature].dropna()
                group_1 = self.df[self.df[target] == 1][feature].dropna()
                
                if len(group_0) > 0 and len(group_1) > 0:
                    t_stat, p_value = stats.ttest_ind(group_0, group_1)
                    effect_size = (group_1.mean() - group_0.mean()) / np.sqrt(
                        (group_0.std()**2 + group_1.std()**2) / 2
                    )
                    
                    results.append({
                        'Feature': feature.replace('_', ' ').title(),
                        'Healthy Mean': round(group_0.mean(), 2),
                        'Diagnosed Mean': round(group_1.mean(), 2),
                        'T-Statistic': round(t_stat, 3),
                        'P-Value': round(p_value, 6),
                        'Effect Size (Cohen\'s d)': round(effect_size, 3),
                        'Significant': '✅' if p_value < 0.05 else '❌'
                    })
        
        return pd.DataFrame(results).sort_values('P-Value')
    
    def plot_comorbidity_analysis(self):
        """Analyze co-occurrence of diseases."""
        existing_targets = [t for t in self.target_cols if t in self.df.columns]
        
        if len(existing_targets) < 2:
            return None
        
        # Co-occurrence matrix
        cooccurrence = self.df[existing_targets].corr()
        
        # Venn-style analysis
        combinations = {}
        for i, t1 in enumerate(existing_targets):
            for j, t2 in enumerate(existing_targets):
                if i < j:
                    both = ((self.df[t1] == 1) & (self.df[t2] == 1)).sum()
                    combinations[f"{t1.replace('_', ' ').title()} + {t2.replace('_', ' ').title()}"] = both
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Disease Correlation', 'Comorbidity Counts'],
            specs=[[{'type': 'heatmap'}, {'type': 'bar'}]]
        )
        
        fig.add_trace(
            go.Heatmap(
                z=cooccurrence.values,
                x=[t.replace('_', ' ').title() for t in cooccurrence.columns],
                y=[t.replace('_', ' ').title() for t in cooccurrence.columns],
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(cooccurrence.values, 3),
                texttemplate='%{text}',
                showscale=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=list(combinations.keys()),
                y=list(combinations.values()),
                marker_color=self.colors['palette'][:len(combinations)],
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="🔗 Disease Comorbidity Analysis",
            height=450,
            template='plotly_white'
        )
        return fig
