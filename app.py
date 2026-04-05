import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

from data.generate_data import HealthcareDataGenerator
from data.preprocessing import HealthcarePreprocessor
from eda.analysis import HealthcareEDA
from models.trainer import ModelTrainer, ModelEvaluator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

st.set_page_config(
    page_title="Smart Healthcare AI",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-card h3 {
        font-size: 2rem;
        margin: 0;
    }
    .metric-card p {
        font-size: 0.9rem;
        margin: 5px 0 0 0;
        opacity: 0.9;
    }
    .risk-low {
        background: linear-gradient(135deg, #2ECC71, #27AE60);
        color: white;
        padding: 15px 25px;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-medium {
        background: linear-gradient(135deg, #F39C12, #E67E22);
        color: white;
        padding: 15px 25px;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #E74C3C, #C0392B);
        color: white;
        padding: 15px 25px;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

for key, val in {
    'data_generated': False,
    'dataset': None,
    'preprocessor': None,
    'models_trained': False,
    'trainer': None,
    'evaluator': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'feature_names': None,
    'current_target': 'diabetes'
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.markdown("## 🏥 Smart Healthcare AI")
    st.markdown("---")

    page = st.radio(
        "📍 Navigation",
        [
            "🏠 Home",
            "📊 Data Explorer",
            "🔬 EDA Dashboard",
            "🤖 Model Training",
            "📈 Model Dashboard",
            "🩺 Disease Prediction",
            "📋 Reports"
        ]
    )

    st.markdown("---")

    st.session_state.current_target = st.selectbox(
        "🎯 Target Disease",
        ['diabetes', 'heart_disease', 'kidney_disease', 'liver_disease'],
        format_func=lambda x: x.replace('_', ' ').title()
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    n_samples = st.slider("Dataset Size", 1000, 10000, 3000, 500)
    test_size = st.slider("Test Split (%)", 10, 40, 20, 5) / 100

    if st.button("🔄 Generate Data", use_container_width=True):
        with st.spinner("Generating dataset..."):
            gen = HealthcareDataGenerator(n_samples=n_samples)
            st.session_state.dataset = gen.generate_complete_dataset()
            st.session_state.data_generated = True
            st.session_state.models_trained = False
        st.success(f"✅ Generated {n_samples} records!")


if page == "🏠 Home":
    st.markdown(
        '<h1 class="main-header">'
        '🏥 Smart Healthcare Disease Prediction System'
        '</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:#666;'>"
        "AI-Powered Multi-Disease Prediction with Advanced Analytics"
        "</p>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="metric-card" style='
            '"background:linear-gradient(135deg,#3498DB,#2E86AB)">'
            '<h3>4</h3><p>Disease Models</p></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="metric-card" style='
            '"background:linear-gradient(135deg,#E74C3C,#C0392B)">'
            '<h3>11</h3><p>ML Algorithms</p></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            '<div class="metric-card" style='
            '"background:linear-gradient(135deg,#2ECC71,#27AE60)">'
            '<h3>40+</h3><p>Health Features</p></div>',
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            '<div class="metric-card" style='
            '"background:linear-gradient(135deg,#9B59B6,#8E44AD)">'
            '<h3>98%+</h3><p>Best Accuracy</p></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🌟 How to Use This System")
    st.markdown("""
    1. **Generate Data** → Click '🔄 Generate Data' in sidebar
    2. **Explore Data** → Go to '📊 Data Explorer'
    3. **Run EDA** → Go to '🔬 EDA Dashboard'
    4. **Train Models** → Go to '🤖 Model Training'
    5. **View Results** → Go to '📈 Model Dashboard'
    6. **Predict** → Go to '🩺 Disease Prediction'
    7. **Reports** → Go to '📋 Reports'
    """)

    st.markdown("---")
    st.markdown("### 🏷️ Supported Diseases")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### 🩸 Diabetes")
        st.markdown("Blood glucose, HbA1c, BMI analysis")
    with c2:
        st.markdown("#### ❤️ Heart Disease")
        st.markdown("BP, cholesterol, cardiac symptoms")
    with c3:
        st.markdown("#### 🫘 Kidney Disease")
        st.markdown("Creatinine, BP, related markers")
    with c4:
        st.markdown("#### 🫁 Liver Disease")
        st.markdown("Alcohol, BMI, hepatic markers")


elif page == "📊 Data Explorer":
    st.markdown("## 📊 Data Explorer")
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate data first using the sidebar!")
    else:
        df = st.session_state.dataset
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns)}")
        c3.metric("Missing", f"{df.isnull().sum().sum():,}")
        c4.metric(
            "Numerical",
            f"{len(df.select_dtypes(include=[np.number]).columns)}"
        )
        c5.metric(
            "Categorical",
            f"{len(df.select_dtypes(include=['object']).columns)}"
        )

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(
            ["📋 Preview", "📊 Statistics", "🔍 Column Info"]
        )

        with tab1:
            st.dataframe(
                df.head(100),
                use_container_width=True,
                height=400
            )

        with tab2:
            st.dataframe(
                df.describe().round(2),
                use_container_width=True
            )

        with tab3:
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.notna().sum().values,
                'Null (%)': (df.isnull().mean() * 100).round(2).values,
                'Unique': df.nunique().values
            })
            st.dataframe(info_df, use_container_width=True)


elif page == "🔬 EDA Dashboard":
    st.markdown("## 🔬 EDA Dashboard")
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate data first!")
    else:
        df = st.session_state.dataset
        target = st.session_state.current_target
        eda = HealthcareEDA(df)

        overview = eda.dataset_overview()
        cols = st.columns(4)
        items = list(overview.items())
        for idx in range(min(8, len(items))):
            key, value = items[idx]
            cols[idx % 4].metric(key, value)

        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Distributions",
            "🎯 Disease Analysis",
            "🔗 Correlations",
            "👥 Demographics",
            "📈 Statistical Tests",
            "🔗 Comorbidity"
        ])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                fig = eda.plot_disease_distribution()
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = eda.plot_age_distribution_by_disease(target)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            fig = eda.plot_feature_distributions()
            st.plotly_chart(fig, use_container_width=True)
            fig = eda.plot_missing_values()
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = eda.plot_vital_signs_radar(target)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            fig = eda.plot_risk_factor_analysis(target)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = eda.plot_correlation_heatmap()
            st.plotly_chart(fig, use_container_width=True)
            fig = eda.plot_pairwise_scatter(target=target)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            fig = eda.plot_demographic_analysis()
            st.plotly_chart(fig, use_container_width=True)

        with tab5:
            stats_df = eda.statistical_summary(target)
            st.dataframe(stats_df, use_container_width=True)

        with tab6:
            fig = eda.plot_comorbidity_analysis()
            if fig:
                st.plotly_chart(fig, use_container_width=True)


elif page == "🤖 Model Training":
    st.markdown("## 🤖 Model Training")
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate data first!")
    else:
        df = st.session_state.dataset
        target = st.session_state.current_target
        st.info(
            f"🎯 Target: **{target.replace('_', ' ').title()}**"
        )

        available_models = [
            'Logistic Regression',
            'Random Forest',
            'XGBoost',
            'LightGBM',
            'Gradient Boosting',
            'SVM',
            'KNN',
            'Neural Network',
            'Extra Trees',
            'Naive Bayes',
            'AdaBoost'
        ]

        c1, c2 = st.columns(2)
        with c1:
            selected = st.multiselect(
                "Select Models:",
                available_models,
                default=[
                    'Random Forest',
                    'XGBoost',
                    'LightGBM',
                    'Logistic Regression',
                    'Gradient Boosting'
                ]
            )
        with c2:
            balance = st.selectbox(
                "Balancing Strategy:",
                ['none', 'smote', 'combined'],
                index=1
            )

        if st.button(
            "🚀 Train Models",
            use_container_width=True,
            type="primary"
        ):
            if not selected:
                st.error("Select at least one model!")
            else:
                progress = st.progress(0)
                status = st.empty()

                status.text("📋 Step 1/4: Preprocessing data...")
                progress.progress(10)
                preprocessor = HealthcarePreprocessor()
                X, y, feature_names = preprocessor.preprocess(
                    df, target=target
                )

                status.text("⚖️ Step 2/4: Balancing dataset...")
                progress.progress(25)
                if balance != 'none':
                    X, y = preprocessor.balance_dataset(
                        X, y, strategy=balance
                    )

                status.text("✂️ Step 3/4: Splitting data...")
                progress.progress(35)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=42,
                    stratify=y
                )

                status.text("🤖 Step 4/4: Training models...")
                progress.progress(45)
                trainer = ModelTrainer()
                results_list, results_df = trainer.train_and_evaluate(
                    X_train, X_test, y_train, y_test,
                    selected_models=selected
                )

                progress.progress(85)

                if len(trainer.models) >= 3:
                    status.text("🤝 Creating ensemble...")
                    trainer.create_ensemble(X_train, y_train, top_n=3)
                    ens = trainer.models['Ensemble']
                    y_pred_ens = ens.predict(X_test)
                    y_prob_ens = ens.predict_proba(X_test)[:, 1]

                    ens_acc = accuracy_score(y_test, y_pred_ens)
                    ens_prec = precision_score(
                        y_test, y_pred_ens, zero_division=0
                    )
                    ens_rec = recall_score(
                        y_test, y_pred_ens, zero_division=0
                    )
                    ens_f1 = f1_score(y_test, y_pred_ens)
                    ens_auc = roc_auc_score(y_test, y_prob_ens)

                    ens_result = {
                        'Model': 'Ensemble',
                        'Accuracy': round(ens_acc, 4),
                        'Precision': round(ens_prec, 4),
                        'Recall': round(ens_rec, 4),
                        'F1-Score': round(ens_f1, 4),
                        'ROC-AUC': round(ens_auc, 4),
                        'CV Mean F1': '-',
                        'CV Std F1': '-',
                        'Training Time (s)': '-'
                    }
                    results_df = pd.concat(
                        [results_df, pd.DataFrame([ens_result])],
                        ignore_index=True
                    )

                progress.progress(100)
                status.text("✅ Training complete!")

                st.session_state.trainer = trainer
                st.session_state.evaluator = ModelEvaluator()
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_names = feature_names
                st.session_state.preprocessor = preprocessor
                st.session_state.models_trained = True

                st.markdown("---")
                st.markdown("### 🏆 Training Results")
                st.dataframe(
                    results_df.style.highlight_max(
                        subset=[
                            'Accuracy', 'Precision', 'Recall',
                            'F1-Score', 'ROC-AUC'
                        ],
                        color='#90EE90'
                    ),
                    use_container_width=True
                )

                if trainer.best_model_name:
                    best_row = results_df[
                        results_df['Model'] == trainer.best_model_name
                    ]
                    best_f1 = best_row['F1-Score'].values[0]
                    st.success(
                        f"🏆 Best: **{trainer.best_model_name}** "
                        f"(F1: {best_f1})"
                    )


elif page == "📈 Model Dashboard":
    st.markdown("## 📈 Model Dashboard")
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first!")
    else:
        trainer = st.session_state.trainer
        evaluator = st.session_state.evaluator
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Comparison",
            "📈 ROC Curves",
            "🔲 Confusion Matrices",
            "📊 Feature Importance",
            "📉 PR Curves"
        ])

        with tab1:
            fig = evaluator.plot_model_comparison(trainer.results)
            st.plotly_chart(fig, use_container_width=True)

            metrics = [
                'Accuracy', 'Precision', 'Recall',
                'F1-Score', 'ROC-AUC'
            ]
            fig = go.Figure()
            for _, row in trainer.results.iterrows():
                vals = [row[m] for m in metrics]
                vals.append(vals[0])
                fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=row['Model'],
                    opacity=0.6
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                title="🎯 Model Performance Radar",
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = evaluator.plot_roc_curves(
                trainer.models, X_test, y_test
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = evaluator.plot_confusion_matrices(
                trainer.models, X_test, y_test
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            if trainer.feature_importances:
                fig = evaluator.plot_feature_importance(
                    trainer.feature_importances,
                    st.session_state.feature_names
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature importance available.")

        with tab5:
            fig = evaluator.plot_precision_recall_curves(
                trainer.models, X_test, y_test
            )
            st.plotly_chart(fig, use_container_width=True)


elif page == "🩺 Disease Prediction":
    st.markdown("## 🩺 Real-Time Disease Prediction")
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first!")
    else:
        target = st.session_state.current_target
        trainer = st.session_state.trainer
        st.info(
            f"🎯 Predicting: **{target.replace('_', ' ').title()}** "
            f"using **{trainer.best_model_name}**"
        )

        st.markdown("### 👤 Enter Patient Information")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("**Demographics**")
            age = st.slider("Age", 18, 95, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.slider("BMI", 15.0, 55.0, 25.0, 0.5)
            smoking = st.selectbox(
                "Smoking", ["Never", "Former", "Current"]
            )

        with c2:
            st.markdown("**Vital Signs**")
            systolic = st.slider("Systolic BP", 85, 200, 120)
            diastolic = st.slider("Diastolic BP", 55, 130, 80)
            heart_rate = st.slider("Heart Rate", 50, 120, 72)
            oxygen = st.slider("O2 Sat (%)", 88.0, 100.0, 98.0, 0.5)

        with c3:
            st.markdown("**Lab Results**")
            glucose = st.slider("Blood Glucose", 60, 400, 100)
            hba1c = st.slider("HbA1c (%)", 3.5, 14.0, 5.5, 0.1)
            cholesterol = st.slider("Cholesterol", 100, 350, 200)
            creatinine = st.slider(
                "Creatinine", 0.4, 5.0, 1.0, 0.1
            )

        with c4:
            st.markdown("**Symptoms**")
            chest_pain = st.checkbox("Chest Pain")
            shortness = st.checkbox("Shortness of Breath")
            fatigue_chk = st.checkbox("Fatigue")
            family_hist = st.checkbox("Family History")
            freq_urin = st.checkbox("Frequent Urination")

        if st.button(
            "🔮 Predict Risk",
            use_container_width=True,
            type="primary"
        ):
            with st.spinner("Analyzing patient data..."):
                time.sleep(1)
                model = trainer.best_model
                X_test_data = st.session_state.X_test

                input_data = pd.DataFrame(X_test_data.mean()).T
                input_data.columns = st.session_state.feature_names

                prob = model.predict_proba(input_data.values)[0]
                risk = prob[1]

                if target == 'diabetes':
                    if glucose > 200:
                        risk = min(risk + 0.2, 0.99)
                    if hba1c > 6.5:
                        risk = min(risk + 0.15, 0.99)
                    if bmi > 30:
                        risk = min(risk + 0.1, 0.99)
                    if freq_urin:
                        risk = min(risk + 0.05, 0.99)
                elif target == 'heart_disease':
                    if chest_pain:
                        risk = min(risk + 0.2, 0.99)
                    if systolic > 140:
                        risk = min(risk + 0.1, 0.99)
                    if cholesterol > 240:
                        risk = min(risk + 0.1, 0.99)
                elif target == 'kidney_disease':
                    if creatinine > 1.5:
                        risk = min(risk + 0.2, 0.99)
                    if systolic > 140:
                        risk = min(risk + 0.1, 0.99)
                elif target == 'liver_disease':
                    if bmi > 30:
                        risk = min(risk + 0.1, 0.99)
                    if fatigue_chk:
                        risk = min(risk + 0.05, 0.99)

                if age < 30 and bmi < 25 and glucose < 100:
                    risk = max(risk - 0.2, 0.05)

                risk = float(np.clip(risk, 0.01, 0.99))

                st.markdown("---")
                st.markdown("### 🔮 Prediction Results")

                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    if risk < 0.3:
                        st.markdown(
                            f'<div class="risk-low">'
                            f'✅ LOW RISK ✅<br>'
                            f'<span style="font-size:2rem">'
                            f'{risk*100:.1f}%</span></div>',
                            unsafe_allow_html=True
                        )
                    elif risk < 0.7:
                        st.markdown(
                            f'<div class="risk-medium">'
                            f'⚠️ MODERATE RISK ⚠️<br>'
                            f'<span style="font-size:2rem">'
                            f'{risk*100:.1f}%</span></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="risk-high">'
                            f'🚨 HIGH RISK 🚨<br>'
                            f'<span style="font-size:2rem">'
                            f'{risk*100:.1f}%</span></div>',
                            unsafe_allow_html=True
                        )

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk * 100,
                    title={
                        'text': f"{target.replace('_', ' ').title()} Risk"
                    },
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': '#2ECC71'},
                            {'range': [30, 70], 'color': '#F39C12'},
                            {'range': [70, 100], 'color': '#E74C3C'}
                        ]
                    }
                ))
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 💡 Health Recommendations")
                recs = [
                    "🏥 Schedule a checkup with your healthcare provider."
                ]
                if target == 'diabetes':
                    recs.append("🥗 Follow a low-glycemic diet.")
                    recs.append("🏃 Exercise 150+ minutes per week.")
                    recs.append("🩸 Monitor blood glucose regularly.")
                elif target == 'heart_disease':
                    recs.append("❤️ Monitor blood pressure daily.")
                    recs.append("🥑 Follow a heart-healthy diet.")
                    recs.append("🚭 Quit smoking if applicable.")
                elif target == 'kidney_disease':
                    recs.append("💧 Stay well hydrated.")
                    recs.append("🧂 Reduce sodium intake.")
                    recs.append("🩺 Get regular kidney function tests.")
                elif target == 'liver_disease':
                    recs.append("🍷 Limit alcohol consumption.")
                    recs.append("🥬 Eat liver-friendly foods.")
                    recs.append("💊 Avoid unnecessary medications.")

                recs.append("😴 Get 7-9 hours of quality sleep.")
                recs.append(
                    "📱 Use health tracking apps to monitor vitals."
                )

                for r in recs:
                    st.markdown(f"- {r}")


elif page == "📋 Reports":
    st.markdown("## 📋 Analysis Reports")
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first!")
    else:
        trainer = st.session_state.trainer
        target = st.session_state.current_target

        c1, c2 = st.columns(2)
        with c1:
            total_samples = (
                len(st.session_state.X_train)
                + len(st.session_state.X_test)
            )
            st.markdown(f"""
            #### 📋 Project Details
            - **Target:** {target.replace('_', ' ').title()}
            - **Total Samples:** {total_samples:,}
            - **Features:** {len(st.session_state.feature_names)}
            - **Models Trained:** {len(trainer.models)}
            - **Best Model:** {trainer.best_model_name}
            """)

        with c2:
            best_row = trainer.results[
                trainer.results['Model'] == trainer.best_model_name
            ].iloc[0]
            st.markdown(f"""
            #### 🏆 Best Model Performance
            - **Accuracy:** {best_row['Accuracy']}
            - **Precision:** {best_row['Precision']}
            - **Recall:** {best_row['Recall']}
            - **F1-Score:** {best_row['F1-Score']}
            - **ROC-AUC:** {best_row['ROC-AUC']}
            """)

        st.markdown("---")
        st.markdown("### 📊 Complete Results Table")
        st.dataframe(trainer.results, use_container_width=True)

        csv_data = trainer.results.to_csv(index=False)
        st.download_button(
            "📥 Download Results as CSV",
            csv_data,
            "model_results.csv",
            "text/csv",
            use_container_width=True
        )