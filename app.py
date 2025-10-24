"""
AI Data Advisor - Complete Dataset Analysis Tool
Versie: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import io
import json

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix
)

api_key = st.secrets["ANTHROPIC_API_KEY"]
with st.expander("🐛 Debug"):
    st.write("Session State:", st.session_state)
    st.write("DataFrame info:", df.info())

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Data Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MAX_FILE_SIZE_MB = 200
SAMPLE_SIZE = 50000

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_bytes(bytes_size: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file, file_type: str) -> Tuple[pd.DataFrame, Dict]:
    """Load dataset with automatic encoding detection."""
    metadata = {
        'filename': uploaded_file.name,
        'file_type': file_type,
        'load_time': None,
        'warnings': []
    }
    
    start_time = datetime.now()
    
    try:
        if file_type == 'csv':
            # Try multiple encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    metadata['encoding'] = encoding
                    break
                except UnicodeDecodeError:
                    continue
        
        elif file_type == 'xlsx':
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            metadata['encoding'] = 'Excel binary'
        
        elif file_type == 'json':
            uploaded_file.seek(0)
            df = pd.read_json(uploaded_file)
            metadata['encoding'] = 'utf-8'
        
        # Sample large datasets
        if len(df) > SAMPLE_SIZE:
            metadata['warnings'].append(
                f"Dataset bevat {len(df):,} rijen. Gesampled naar {SAMPLE_SIZE:,} voor performance."
            )
            df = df.sample(n=SAMPLE_SIZE, random_state=42)
        
        metadata['load_time'] = (datetime.now() - start_time).total_seconds()
        return df, metadata
    
    except Exception as e:
        st.error(f"❌ Fout bij laden: {str(e)}")
        return None, metadata

def detect_problem_type(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect whether data is time series, regression, or classification."""
    result = {
        'type': 'unknown',
        'confidence': 0.0,
        'date_columns': [],
        'potential_targets': [],
        'reasoning': []
    }
    
    # Detect date columns
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            date_cols.append(col)
        elif df[col].dtype == 'object':
            try:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() / len(sample) > 0.8:
                        date_cols.append(col)
            except:
                pass
    
    result['date_columns'] = date_cols
    
    # Detect potential targets
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in numeric_cols:
        if df[col].nunique() / len(df) < 0.95:  # Not an ID
            result['potential_targets'].append({
                'column': col,
                'type': 'regression',
                'cardinality': df[col].nunique()
            })
    
    for col in categorical_cols:
        card = df[col].nunique()
        if 2 <= card <= 50:
            result['potential_targets'].append({
                'column': col,
                'type': 'classification',
                'cardinality': card
            })
    
    # Determine problem type
    if len(date_cols) > 0:
        result['type'] = 'time_series'
        result['confidence'] = 0.85
        result['reasoning'].append(f"Datumkolom(men) gedetecteerd: {date_cols}")
    elif len(result['potential_targets']) > 0:
        first_target = result['potential_targets'][0]
        result['type'] = first_target['type']
        result['confidence'] = 0.7
        result['reasoning'].append(f"Potentiële target: {first_target['column']}")
    else:
        result['type'] = 'unsupervised'
        result['confidence'] = 0.6
        result['reasoning'].append("Geen duidelijke target - focus op EDA")
    
    return result

@st.cache_data(show_spinner=False)
def build_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Build comprehensive EDA summary."""
    summary = {
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing': {},
        'duplicates': df.duplicated().sum(),
        'numeric_summary': {},
        'categorical_summary': {},
        'correlations': []
    }
    
    # Missing values
    missing = df.isnull().sum()
    summary['missing'] = {
        col: {
            'count': int(val),
            'percentage': float(val / len(df) * 100)
        }
        for col, val in missing.items() if val > 0
    }
    
    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:20]:
        summary['numeric_summary'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median())
        }
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols[:20]:
        summary['categorical_summary'][col] = {
            'unique': int(df[col].nunique()),
            'top_value': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None
        }
    
    # Top correlations
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols[:20]].corr()
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'col1': corr_matrix.columns[i],
                    'col2': corr_matrix.columns[j],
                    'correlation': float(corr_matrix.iloc[i, j])
                })
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        summary['correlations'] = corr_pairs[:10]
    
    return summary

# ============================================================================
# ML PIPELINES
# ============================================================================

def regression_pipeline(df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Dict:
    """Simple regression pipeline with Random Forest."""
    results = {'success': False, 'error': None, 'metrics': {}, 'figures': {}}
    
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Only numeric features
        X_numeric = X.select_dtypes(include=[np.number]).fillna(X.select_dtypes(include=[np.number]).median())
        
        if X_numeric.shape[1] == 0:
            results['error'] = "Geen numerieke features"
            return results
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42
        )
        
        # Train model
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Metrics
        results['metrics']['Random Forest'] = {
            'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'MAE': float(mean_absolute_error(y_test, y_pred)),
            'R²': float(r2_score(y_test, y_pred))
        }
        
        # Actual vs Predicted plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Actual vs Predicted',
            xaxis_title='Actual',
            yaxis_title='Predicted',
            height=400
        )
        
        results['figures']['actual_vs_pred'] = fig
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': X_numeric.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        fig2 = go.Figure(go.Bar(
            x=importances['importance'],
            y=importances['feature'],
            orientation='h',
            marker=dict(color='steelblue')
        ))
        fig2.update_layout(title='Feature Importance', height=400)
        
        results['figures']['feature_importance'] = fig2
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def classification_pipeline(df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Dict:
    """Simple classification pipeline with Random Forest."""
    results = {'success': False, 'error': None, 'metrics': {}, 'figures': {}}
    
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode target if needed
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Only numeric features
        X_numeric = X.select_dtypes(include=[np.number]).fillna(X.select_dtypes(include=[np.number]).median())
        
        if X_numeric.shape[1] == 0:
            results['error'] = "Geen numerieke features"
            return results
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Metrics
        results['metrics']['Random Forest'] = {
            'Accuracy': float(accuracy_score(y_test, y_pred)),
            'F1': float(f1_score(y_test, y_pred, average='weighted'))
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}'
        ))
        fig.update_layout(title='Confusion Matrix', height=400)
        
        results['figures']['confusion_matrix'] = fig
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def generate_advice(summary: Dict[str, Any]) -> str:
    """Generate rule-based data advice."""
    advice = ["## 🎯 Data Advies\n"]
    
    # Data Quality
    advice.append("### 1️⃣ Data Kwaliteit\n")
    
    if summary['missing']:
        advice.append(f"- **Missing values**: {len(summary['missing'])} kolommen")
        advice.append("  - Numeriek: median/mean imputation")
        advice.append("  - Categorisch: mode of 'unknown' categorie\n")
    
    if summary['duplicates'] > 0:
        advice.append(f"- **Duplicaten**: {summary['duplicates']:,} rijen")
        advice.append("  - Verwijder na verificatie\n")
    
    # Feature Engineering
    advice.append("### 2️⃣ Feature Engineering\n")
    
    if summary['correlations']:
        top_corr = summary['correlations'][0]
        advice.append(f"- Hoogste correlatie: {top_corr['col1']} ↔ {top_corr['col2']} ({top_corr['correlation']:.3f})")
        advice.append("  - Check multicollineariteit\n")
    
    # Modeling
    advice.append("### 3️⃣ Modellering\n")
    advice.append("- Start met baseline modellen")
    advice.append("- Random Forest voor eerste iteratie")
    advice.append("- Cross-validatie voor robuuste evaluatie\n")
    
    # Next Steps
    advice.append("### 4️⃣ Volgende Stappen\n")
    advice.append("1. Diepgaande EDA per feature")
    advice.append("2. Feature engineering met domeinkennis")
    advice.append("3. Hyperparameter tuning")
    advice.append("4. Model ensemble overwegen")
    
    return "\n".join(advice)

# ============================================================================
# DEMO DATASET
# ============================================================================

@st.cache_data
def generate_demo_dataset() -> pd.DataFrame:
    """Generate synthetic demo dataset."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': 1000 + np.cumsum(np.random.randn(n) * 10) + 200 * np.sin(2 * np.pi * np.arange(n) / 365),
        'Temperature': np.random.uniform(5, 35, n),
        'Promotion': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'DayOfWeek': dates.dayofweek
    })
    
    return df

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Data Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#666;">Intelligente dataset-analyse met ML en EDA</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Laden")
        
        use_demo = st.checkbox("🎓 Gebruik demo dataset")
        
        if use_demo:
            df = generate_demo_dataset()
            metadata = {'filename': 'demo.csv', 'warnings': []}
            st.success("✅ Demo geladen")
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV, XLSX of JSON",
                type=['csv', 'xlsx', 'json']
            )
            
            if uploaded_file:
                file_type = uploaded_file.name.split('.')[-1]
                with st.spinner("Laden..."):
                    df, metadata = load_dataset(uploaded_file, file_type)
                
                if df is not None:
                    st.success(f"✅ {metadata['filename']}")
                    for warning in metadata['warnings']:
                        st.warning(warning)
            else:
                st.info("👆 Upload een bestand")
                st.stop()
        
        st.markdown("---")
        st.markdown("### 📊 Statistieken")
        st.metric("Rijen", f"{len(df):,}")
        st.metric("Kolommen", len(df.columns))
    
    # Main Content
    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overzicht", "🔍 EDA", "🤖 Modellen", "💡 Advies"])
        
        # TAB 1: OVERVIEW
        with tab1:
            st.markdown("## 📋 Dataset Overzicht")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Rijen", f"{len(df):,}")
            col2.metric("📈 Kolommen", len(df.columns))
            col3.metric("💾 Geheugen", format_bytes(df.memory_usage(deep=True).sum()))
            col4.metric("📝 Duplicaten", f"{df.duplicated().sum():,}")
            
            st.markdown("---")
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("### 🎯 Probleem Detectie")
            problem_info = detect_problem_type(df)
            
            st.markdown(f"""
            <div class="success-box">
                <h4>Gedetecteerd: {problem_info['type'].upper()}</h4>
                <p><strong>Confidence:</strong> {problem_info['confidence']*100:.0f}%</p>
                <p><strong>Redenering:</strong> {', '.join(problem_info['reasoning'])}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # TAB 2: EDA
        with tab2:
            st.markdown("## 🔍 Exploratory Data Analysis")
            
            with st.spinner("Analyseren..."):
                eda_summary = build_eda_summary(df)
            
            # Missing values
            st.markdown("### 🕳️ Missing Values")
            if eda_summary['missing']:
                missing_df = pd.DataFrame([
                    {'Kolom': col, 'Missing %': f"{info['percentage']:.2f}%"}
                    for col, info in eda_summary['missing'].items()
                ])
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ Geen missing values!")
            
            st.markdown("---")
            
            # Numeric distributions
            st.markdown("### 📊 Numerieke Kolommen")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Selecteer kolom", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(df, x=selected_col, nbins=50, title=f"Distributie: {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown(f"**Statistieken**")
                    st.dataframe(df[selected_col].describe())
            
            st.markdown("---")
            
            # Correlations
            st.markdown("### 🔗 Correlaties")
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols[:10]].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 3: MODELS
        with tab3:
            st.markdown("## 🤖 Machine Learning")
            
            problem_info = detect_problem_type(df)
            
            col1, col2 = st.columns(2)
            with col1:
                problem_type = st.selectbox(
                    "Probleem Type",
                    ['regression', 'classification'],
                    index=0 if problem_info['type'] == 'regression' else 1
                )
            
            with col2:
                targets = [t['column'] for t in problem_info['potential_targets']]
                if targets:
                    target_col = st.selectbox("Target Kolom", targets)
                else:
                    target_col = st.selectbox("Target Kolom", df.columns)
            
            if st.button("🚀 Train Model", type="primary"):
                if problem_type == 'regression':
                    with st.spinner("Training..."):
                        results = regression_pipeline(df, target_col)
                    
                    if results['success']:
                        st.success("✅ Model getraind!")
                        
                        st.markdown("### 📊 Metrics")
                        metrics_df = pd.DataFrame(results['metrics']).T
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(results['figures']['actual_vs_pred'], use_container_width=True)
                        with col2:
                            st.plotly_chart(results['figures']['feature_importance'], use_container_width=True)
                    else:
                        st.error(f"❌ Fout: {results['error']}")
                
                else:  # classification
                    with st.spinner("Training..."):
                        results = classification_pipeline(df, target_col)
                    
                    if results['success']:
                        st.success("✅ Model getraind!")
                        
                        st.markdown("### 📊 Metrics")
                        metrics_df = pd.DataFrame(results['metrics']).T
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        st.plotly_chart(results['figures']['confusion_matrix'], use_container_width=True)
                    else:
                        st.error(f"❌ Fout: {results['error']}")
        
        # TAB 4: ADVICE
        with tab4:
            st.markdown("## 💡 Data Advies")
            
            if st.button("🤖 Genereer Advies", type="primary"):
                with st.spinner("Analyseren..."):
                    eda_summary = build_eda_summary(df)
                    advice = generate_advice(eda_summary)
                
                st.markdown(advice)
                
                st.download_button(
                    "📥 Download Advies",
                    data=advice,
                    file_name="data_advies.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()