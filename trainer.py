import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import time
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = pd.DataFrame()
        self.best_model = None
        self.best_model_name = None
        self.feature_importances = {}

    def get_models(self):
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                n_jobs=-1
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.random_state
            )
        }

    def train_and_evaluate(
        self, X_train, X_test, y_train, y_test,
        selected_models=None
    ):
        all_models = self.get_models()

        if selected_models:
            models_to_train = {
                k: v for k, v in all_models.items()
                if k in selected_models
            }
        else:
            models_to_train = all_models

        results = []

        for name, model in models_to_train.items():
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time

                y_pred = model.predict(X_test)

                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = None

                cv = StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=self.random_state
                )
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring='f1'
                )

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(
                    y_test, y_pred, zero_division=0
                )
                rec = recall_score(
                    y_test, y_pred, zero_division=0
                )
                f1 = f1_score(y_test, y_pred, zero_division=0)

                if y_prob is not None:
                    auc = roc_auc_score(y_test, y_prob)
                else:
                    auc = 0

                if hasattr(model, 'feature_importances_'):
                    self.feature_importances[name] = (
                        model.feature_importances_
                    )
                elif hasattr(model, 'coef_'):
                    self.feature_importances[name] = (
                        np.abs(model.coef_[0])
                    )

                results.append({
                    'Model': name,
                    'Accuracy': round(acc, 4),
                    'Precision': round(prec, 4),
                    'Recall': round(rec, 4),
                    'F1-Score': round(f1, 4),
                    'ROC-AUC': round(auc, 4),
                    'CV Mean F1': round(cv_scores.mean(), 4),
                    'CV Std F1': round(cv_scores.std(), 4),
                    'Training Time (s)': round(train_time, 3)
                })
                self.models[name] = model

            except Exception as e:
                print(f"Error training {name}: {e}")

        self.results = pd.DataFrame(results)

        if len(self.results) > 0:
            best_idx = self.results['F1-Score'].idxmax()
            self.best_model_name = self.results.loc[
                best_idx, 'Model'
            ]
            self.best_model = self.models[self.best_model_name]

        return results, self.results

    def create_ensemble(self, X_train, y_train, top_n=3):
        if len(self.results) == 0:
            return None

        top_models = self.results.nlargest(
            top_n, 'F1-Score'
        )['Model'].tolist()

        estimators = [
            (name, self.models[name])
            for name in top_models
            if name in self.models
        ]

        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        ensemble.fit(X_train, y_train)
        self.models['Ensemble'] = ensemble
        return ensemble


class ModelEvaluator:
    def __init__(self):
        self.colors = px.colors.qualitative.Set2

    def plot_model_comparison(self, results_df):
        fig = go.Figure()

        metrics = [
            'Accuracy', 'Precision', 'Recall',
            'F1-Score', 'ROC-AUC'
        ]

        for idx, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Model'],
                y=results_df[metric],
                marker_color=self.colors[idx % len(self.colors)],
                text=results_df[metric].round(3),
                textposition='auto'
            ))

        fig.update_layout(
            title="🏆 Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            template='plotly_white',
            height=500,
            xaxis_tickangle=-45,
            yaxis_range=[0, 1.1]
        )
        return fig

    def plot_roc_curves(self, models, X_test, y_test):
        fig = go.Figure()

        for idx, (name, model) in enumerate(models.items()):
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{name} (AUC={auc:.3f})',
                    line=dict(
                        color=self.colors[idx % len(self.colors)]
                    )
                ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title="📈 ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template='plotly_white',
            height=500
        )
        return fig

    def plot_confusion_matrices(self, models, X_test, y_test):
        n = len(models)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=list(models.keys())
        )

        for idx, (name, model) in enumerate(models.items()):
            row = idx // ncols + 1
            col = idx % ncols + 1

            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            cm_sum = cm.sum(axis=1)[:, np.newaxis]
            cm_norm = cm.astype('float') / cm_sum

            text_array = []
            for i in range(2):
                row_text = []
                for j in range(2):
                    cell_text = (
                        f'{cm[i][j]}\n({cm_norm[i][j]:.1%})'
                    )
                    row_text.append(cell_text)
                text_array.append(row_text)

            text_np = np.array(text_array)

            fig.add_trace(
                go.Heatmap(
                    z=cm_norm,
                    x=['Pred Neg', 'Pred Pos'],
                    y=['Act Neg', 'Act Pos'],
                    colorscale='Blues',
                    showscale=False,
                    text=text_np,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ),
                row=row,
                col=col
            )

        fig.update_layout(
            title="🔲 Confusion Matrices",
            height=350 * nrows,
            template='plotly_white'
        )
        return fig

    def plot_feature_importance(
        self, importance_dict, feature_names, top_n=15
    ):
        num_models = len(importance_dict)
        fig = make_subplots(
            rows=1,
            cols=num_models,
            subplot_titles=list(importance_dict.keys())
        )

        for idx, (name, importances) in enumerate(
            importance_dict.items()
        ):
            indices = np.argsort(importances)[-top_n:]
            top_features = [
                feature_names[i].replace('_', ' ').title()
                for i in indices
            ]
            top_values = importances[indices]

            fig.add_trace(
                go.Bar(
                    y=top_features,
                    x=top_values,
                    orientation='h',
                    marker_color=self.colors[
                        idx % len(self.colors)
                    ],
                    showlegend=False
                ),
                row=1,
                col=idx + 1
            )

        fig.update_layout(
            title="📊 Feature Importance",
            height=500,
            template='plotly_white'
        )
        return fig

    def plot_precision_recall_curves(
        self, models, X_test, y_test
    ):
        fig = go.Figure()

        for idx, (name, model) in enumerate(models.items()):
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                prec, rec, _ = precision_recall_curve(
                    y_test, y_prob
                )
                ap = average_precision_score(y_test, y_prob)
                fig.add_trace(go.Scatter(
                    x=rec,
                    y=prec,
                    mode='lines',
                    name=f'{name} (AP={ap:.3f})',
                    line=dict(
                        color=self.colors[idx % len(self.colors)]
                    )
                ))

        fig.update_layout(
            title="📉 Precision-Recall Curves",
            xaxis_title="Recall",
            yaxis_title="Precision",
            template='plotly_white',
            height=500
        )
        return fig