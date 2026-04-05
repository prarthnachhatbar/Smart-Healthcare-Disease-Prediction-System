# models/explainer.py
"""
Model Explainability using SHAP
"""

import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st


class ModelExplainer:
    """SHAP-based model explainability."""
    
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        # Create appropriate explainer
        try:
            self.explainer = shap.TreeExplainer(model)
        except:
            try:
                self.explainer = shap.KernelExplainer(
                    model.predict_proba, 
                    shap.sample(X_train, 100)
                )
            except:
                self.explainer = None
    
    def compute_shap_values(self, X):
        """Compute SHAP values for given data."""
        if self.explainer is not None:
            self.shap_values = self.explainer.shap_values(X)
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # For binary classification
            return self.shap_values
        return None
    
    def plot_global_importance(self, X, top_n=15):
        """Plot global SHAP feature importance."""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        if self.shap_values is None:
            return None
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        indices = np.argsort(mean_abs_shap)[-top_n:]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[self.feature_names[i].replace('_', ' ').title() for i in indices],
            x=mean_abs_shap[indices],
            orientation='h',
            marker_color='#2E86AB'
        ))
        
        fig.update_layout(
            title="🔍 SHAP Global Feature Importance",
            xaxis_title="Mean |SHAP Value|",
            height=500,
            template='plotly_white'
        )
        return fig
    
    def explain_single_prediction(self, X_instance, feature_names):
        """Explain a single prediction using SHAP."""
        if self.explainer is None:
            return None, None
        
        shap_vals = self.explainer.shap_values(X_instance.reshape(1, -1))
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        # Create waterfall data
        shap_vals_flat = shap_vals.flatten()
        
        # Sort by absolute value
        indices = np.argsort(np.abs(shap_vals_flat))[::-1][:10]
        
        features_sorted = [feature_names[i].replace('_', ' ').title() for i in indices]
        values_sorted = shap_vals_flat[indices]
        
        colors = ['#E74C3C' if v > 0 else '#2ECC71' for v in values_sorted]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features_sorted[::-1],
            x=values_sorted[::-1],
            orientation='h',
            marker_color=colors[::-1]
        ))
        
        fig.update_layout(
            title="🎯 Individual Prediction Explanation",
            xaxis_title="SHAP Value (Impact on Prediction)",
            height=400,
            template='plotly_white'
        )
        
        return fig, shap_vals_flat
