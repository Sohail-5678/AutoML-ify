import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Set page configuration
st.set_page_config(
    page_title="AutoML-ify: Your Simplified ML Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2196F3;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .card p {
        color: #333333 !important;
    }
    .metric-card {
        background-color: #e8f4f8;
        border-radius: 8px;
        padding: 15px;
        margin: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-card h3, .metric-card p {
        color: #333333 !important;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0277bd;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #546e7a;
    }
    .feature-importance {
        height: 400px;
        overflow-y: auto;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .step-card {
        background-color: #f1f8e9;
        border-left: 5px solid #7cb342;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .step-card h3, .step-card p {
        color: #333333 !important;
    }
    .model-comparison-table {
        font-size: 0.85rem;
    }
    /* Only target main content area, not the sidebar */
    .main .stMarkdown p, 
    .main .stMarkdown h3, 
    .main .stMarkdown li {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)



# Helper functions
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        else:
            st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def detect_date_columns(df):
    """Detect potential date columns in the dataframe"""
    date_columns = []
    
    for col in df.columns:
        # Check if column name contains date-related keywords
        if any(date_word in col.lower() for date_word in ['date', 'time', 'day', 'month', 'year']):
            date_columns.append(col)
        # Try to convert to datetime to see if it works
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='raise')
                date_columns.append(col)
            except:
                pass
    
    return date_columns

def handle_date_columns(df, date_columns):
    """Ask user how to handle date columns"""
    if not date_columns:
        return df, []
    
    st.write("### Date Columns Detected")
    st.write("The following columns appear to contain date information:")
    
    for col in date_columns:
        st.write(f"- {col}")
    
    date_handling = st.radio(
        "How would you like to handle date columns?",
        ["Use as index (select one)", "Extract date features", "Keep as is", "Drop date columns"]
    )
    
    if date_handling == "Use as index (select one)":
        index_col = st.selectbox("Select date column to use as index", date_columns)
        try:
            df[index_col] = pd.to_datetime(df[index_col])
            df = df.set_index(index_col)
            st.success(f"Set {index_col} as index")
            # Remove the selected column from date_columns list
            date_columns.remove(index_col)
        except Exception as e:
            st.error(f"Error setting index: {e}")
    
    elif date_handling == "Extract date features":
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
                # Add only basic features to avoid too many columns
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                # Drop original column
                df = df.drop(col, axis=1)
                st.success(f"Extracted features from {col}")
            except Exception as e:
                st.warning(f"Could not process {col}: {e}")
    
    elif date_handling == "Drop date columns":
        df = df.drop(date_columns, axis=1)
        st.success(f"Dropped {len(date_columns)} date columns")
    
    return df, date_columns

def clean_data(df, threshold_pct=0.5, handle_categorical=True):
    # Create a copy of the dataframe
    df_cleaned = df.copy()
    
    # Track cleaning steps for reporting
    cleaning_report = []
    
    # Drop columns with too many missing values
    threshold_rows = int(df.shape[0] * threshold_pct)
    cols_before = df_cleaned.shape[1]
    df_cleaned = df_cleaned.dropna(axis=1, thresh=threshold_rows)
    cols_after = df_cleaned.shape[1]
    if cols_before > cols_after:
        cleaning_report.append(f"Dropped {cols_before - cols_after} columns with more than {threshold_pct*100}% missing values.")
    
    # Handle remaining missing values
    numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_cleaned.select_dtypes(exclude=['int64', 'float64']).columns
    
    # For numeric columns, fill missing values with median
    if len(numeric_cols) > 0:
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
        cleaning_report.append(f"Filled missing values in {len(numeric_cols)} numeric columns with median values.")
    
    # For categorical columns, fill missing values with mode
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            if df_cleaned[col].isna().any():
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        cleaning_report.append(f"Filled missing values in categorical columns with mode values.")
    
    # Handle categorical features - One Hot Encoding
    if handle_categorical:
        categorical_cols = df_cleaned.select_dtypes(exclude=['int64', 'float64']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                # Skip if column has too many unique values
                if df_cleaned[col].nunique() < 20:  # Only encode if fewer than 20 unique values
                    # Get dummies and join to dataframe
                    dummies = pd.get_dummies(df_cleaned[col], prefix=col, drop_first=True)
                    df_cleaned = pd.concat([df_cleaned, dummies], axis=1)
                    # Drop original column
                    df_cleaned.drop(col, axis=1, inplace=True)
            cleaning_report.append(f"Applied one-hot encoding to categorical columns with fewer than 20 unique values.")
    
    # Convert all columns to standard types to avoid JSON serialization issues
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype.name not in ['int64', 'float64', 'bool', 'object']:
            df_cleaned[col] = df_cleaned[col].astype('float64')
    
    return df_cleaned, cleaning_report

def generate_eda_plots(df):
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())
    col4.metric("Duplicated Rows", df.duplicated().sum())
    
    # Numeric columns distribution
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) > 0:
        st.subheader("Numeric Features Distribution")
        
        # Create a distribution plot for each numeric column
        for i in range(0, len(numeric_cols), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(numeric_cols):
                    col = numeric_cols[i + j]
                    fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
                    cols[j].plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation")
        st.plotly_chart(fig, use_container_width=True)
    
    # Categorical columns
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    
    if len(categorical_cols) > 0:
        st.subheader("Categorical Features")
        
        for i in range(0, len(categorical_cols), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(categorical_cols):
                    col = categorical_cols[i + j]
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = [col, 'Count']
                    fig = px.bar(value_counts, x=col, y='Count', title=f"Distribution of {col}")
                    cols[j].plotly_chart(fig, use_container_width=True)

def train_and_evaluate_models(X, y, problem_type, test_size=0.25, random_state=42):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    model_objects = {}
    
    if problem_type == "Classification":
        # Define classification models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Support Vector Machine': SVC(probability=True, random_state=random_state),
            'Gaussian Naive Bayes': GaussianNB(),
            'XGBoost': XGBClassifier(random_state=random_state)
        }
        
        with st.spinner('Training and evaluating classification models...'):
            progress_bar = st.progress(0)
            
            for i, (name, model) in enumerate(models.items()):
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Store results
                results[name] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }
                
                model_objects[name] = model
                progress_bar.progress((i + 1) / len(models))
                
            progress_bar.empty()
            
    elif problem_type == "Regression":
        # Define regression models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=random_state),
            'Lasso Regression': Lasso(random_state=random_state),
            'ElasticNet': ElasticNet(random_state=random_state),
            'Decision Tree': DecisionTreeRegressor(random_state=random_state),
            'Random Forest': RandomForestRegressor(random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
            'SVR': SVR()
        }
        
        with st.spinner('Training and evaluating regression models...'):
            progress_bar = st.progress(0)
            
            for i, (name, model) in enumerate(models.items()):
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store results
                results[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R² Score': r2
                }
                
                model_objects[name] = model
                progress_bar.progress((i + 1) / len(models))
                
            progress_bar.empty()
    
    return results, model_objects, X_train_scaled, X_test_scaled, y_train, y_test

def visualize_model_comparison(results, problem_type):
    # Convert results to DataFrame
    df_results = pd.DataFrame(results).T.reset_index()
    df_results.rename(columns={'index': 'Model'}, inplace=True)
    
    # Create comparison visualizations
    if problem_type == "Classification":
        # Metrics to display
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Create bar chart for each metric
        for metric in metrics:
            fig = px.bar(
                df_results, 
                x='Model', 
                y=metric, 
                title=f'Model Comparison - {metric}',
                color=metric,
                color_continuous_scale='viridis',
                text=df_results[metric].apply(lambda x: f'{x:.3f}')
            )
            fig.update_layout(
                xaxis_title='Model',
                yaxis_title=metric,
                yaxis_range=[0, 1],
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart for overall comparison
        fig = go.Figure()
        
        for model in df_results['Model']:
            model_metrics = df_results[df_results['Model'] == model][metrics].values[0]
            fig.add_trace(go.Scatterpolar(
                r=model_metrics,
                theta=metrics,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Model Performance Comparison",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif problem_type == "Regression":
        # Metrics to display
        metrics = ['MSE', 'RMSE', 'MAE', 'R² Score']
        
        # Error metrics (lower is better)
        error_metrics = ['MSE', 'RMSE', 'MAE']
        
        # Create bar charts for error metrics
        for metric in error_metrics:
            fig = px.bar(
                df_results, 
                x='Model', 
                y=metric, 
                title=f'Model Comparison - {metric} (Lower is Better)',
                color=metric,
                color_continuous_scale='reds_r',
                text=df_results[metric].apply(lambda x: f'{x:.3f}')
            )
            fig.update_layout(
                xaxis_title='Model',
                yaxis_title=metric,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Create bar chart for R² Score (higher is better)
        fig = px.bar(
            df_results, 
            x='Model', 
            y='R² Score', 
            title='Model Comparison - R² Score (Higher is Better)',
            color='R² Score',
            color_continuous_scale='viridis',
            text=df_results['R² Score'].apply(lambda x: f'{x:.3f}')
        )
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='R² Score',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def visualize_feature_importance(model, feature_names, model_name):
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create DataFrame for visualization
        df_importance = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Plot feature importance
        fig = px.bar(
            df_importance.head(15), 
            x='Importance', 
            y='Feature',
            orientation='h',
            title=f'Top 15 Feature Importances - {model_name}',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    elif model_name == 'Logistic Regression' or hasattr(model, 'coef_'):
        # For linear models
        coefficients = model.coef_[0] if model_name == 'Logistic Regression' else model.coef_
        
        # Create DataFrame for visualization
        df_coef = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        })
        df_coef = df_coef.reindex(df_coef['Coefficient'].abs().sort_values(ascending=False).index)
        
        # Plot coefficients
        fig = px.bar(
            df_coef.head(15),
            x='Coefficient',
            y='Feature',
            orientation='h',
            title=f'Top 15 Feature Coefficients - {model_name}',
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Feature importance visualization not available for {model_name}")

def visualize_classification_results(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=np.unique(y_test),
        y=np.unique(y_test),
        text_auto=True,
        color_continuous_scale='blues',
        title=f'Confusion Matrix - {model_name}'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve for binary classification
    if len(np.unique(y_test)) == 2:
        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            
            # Plot ROC curve
            fig = px.line(
                x=fpr, y=tpr,
                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                title=f'ROC Curve - {model_name} (AUC = {auc:.3f})'
            )
            
            # Add diagonal line (random classifier)
            fig.add_shape(
                type='line',
                line=dict(dash='dash', color='gray'),
                x0=0, x1=1, y0=0, y1=1
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def visualize_regression_results(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create scatter plot of actual vs predicted values
    fig = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title=f'Actual vs Predicted Values - {model_name}'
    )
    
    # Add diagonal line (perfect predictions)
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='gray'),
        x0=min(y_test), x1=max(y_test),
        y0=min(y_test), y1=max(y_test)
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Residual plot
    residuals = y_test - y_pred
    
    fig = px.scatter(
        x=y_pred, y=residuals,
        labels={'x': 'Predicted Values', 'y': 'Residuals'},
        title=f'Residual Plot - {model_name}'
    )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='gray'),
        x0=min(y_pred), x1=max(y_pred),
        y0=0, y1=0
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Residual distribution
    fig = px.histogram(
        residuals,
        title=f'Residual Distribution - {model_name}',
        labels={'value': 'Residual Value', 'count': 'Frequency'},
        marginal='box'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Main application
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("AutoML-ify")
        choice = st.radio("Navigation", ["Home", "Upload", "EDA", "Data Cleaning", "Modelling", "Download"])
        st.info("This application helps you build and explore your data with automated machine learning.")
    
    # Home page
    if choice == "Home":
        st.markdown('<h1 class="main-header">Welcome to AutoML-ify: Your Simplified ML Pipeline, Automated!</h1>', unsafe_allow_html=True)
        
        # Introduction
        st.markdown("""
        <div class="card">
            <p>AutoML-ify is designed to make the process of analyzing and selecting the best machine learning model for your dataset as easy and efficient as possible. With just a few clicks, you can upload your dataset, clean it, analyze it, train multiple models, and visualize the results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        st.markdown('<h2 class="sub-header">What AutoML-ify Does</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="step-card">
                <h3>📊 Automated Data Cleaning</h3>
                <p>Handles missing values, outliers, and inconsistencies with smart preprocessing techniques.</p>
            </div>
            
            <div class="step-card">
                <h3>🔍 Exploratory Data Analysis (EDA)</h3>
                <p>Generates insightful visualizations and statistics to help you understand your data better.</p>
            </div>
            
            <div class="step-card">
                <h3>🤖 Model Selection & Training</h3>
                <p>Automatically selects and trains multiple models for your task, comparing their performance.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="step-card">
                <h3>📈 Visualization & Reporting</h3>
                <p>Provides clear, interactive visualizations of model performance and data insights.</p>
            </div>
            
            <div class="step-card">
                <h3>💾 Model Export</h3>
                <p>Export your best-performing model for deployment in your applications.</p>
            </div>
            
            <div class="step-card">
                <h3>🔄 End-to-End Pipeline</h3>
                <p>Complete workflow from data upload to model selection in one application.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # How it works
        st.markdown('<h2 class="sub-header">How It Works</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Step 1</h3>
                <p>Upload Your Dataset</p>
                <p>CSV, Excel, or JSON</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Step 2</h3>
                <p>Clean & Analyze</p>
                <p>Automated preprocessing</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Step 3</h3>
                <p>Train Models</p>
                <p>Multiple algorithms</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Step 4</h3>
                <p>Export Results</p>
                <p>Download best model</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Benefits
        st.markdown('<h2 class="sub-header">Why Choose AutoML-ify?</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>⏱️ Save Time</h3>
                <p>Automate repetitive tasks and focus on interpreting results.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔍 No Expertise Required</h3>
                <p>Perfect for users with limited ML knowledge.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Scalable</h3>
                <p>Handles diverse datasets and adapts to various ML tasks.</p>
            </div>
            """, unsafe_allow_html=True)
    
   # Upload page
    elif choice == "Upload":
        st.markdown('<h1 class="main-header">Upload Your Dataset</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <p>Upload your dataset in CSV, Excel, or JSON format. The application will automatically detect the file type and load the data for analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        file = st.file_uploader("Upload Your Dataset", type=["csv", "xlsx", "xls", "json"])
        
        if file:
            with st.spinner('Loading data...'):
                df = load_data(file)
                
                if df is not None:
                    # Check for date columns
                    date_columns = detect_date_columns(df)
                    
                    if date_columns:
                        df, remaining_date_cols = handle_date_columns(df, date_columns)
                    
                    # Convert any non-standard dtypes to standard ones to avoid JSON serialization issues
                    for col in df.columns:
                        if df[col].dtype.name not in ['int64', 'float64', 'bool', 'object']:
                            df[col] = df[col].astype(str)
                    
                    # Save to session state
                    st.session_state['data'] = df
                    st.session_state['filename'] = file.name
                    
                    # Display data info
                    st.markdown('<h2 class="sub-header">Data Preview</h2>', unsafe_allow_html=True)
                    
                    # Display basic information
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", df.shape[0])
                    col2.metric("Columns", df.shape[1])
                    col3.metric("Data Types", len(df.dtypes.unique()))
                    
                    # Show the first few rows of the dataframe
                    st.dataframe(df.head(10))
                    
                    # Display data types
                    st.markdown('<h3>Data Types</h3>', unsafe_allow_html=True)
                    
                    # Create a more visually appealing display of data types
                    dtypes_df = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count().values,
                        'Null Count': df.isna().sum().values,
                        'Null Percentage': (df.isna().sum() / len(df) * 100).round(2).astype(str) + '%'
                    })
                    
                    st.dataframe(dtypes_df, use_container_width=True)
                    
                    # Save the dataframe to a CSV file for later use
                    df.to_csv('dataset.csv', index=None)
                    st.success(f"Successfully loaded {file.name} with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # EDA page
    elif choice == "EDA":
        st.markdown('<h1 class="main-header">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
        
        # Check if data is loaded
        if 'data' not in st.session_state:
            st.warning("Please upload your dataset first.")
            return
        
        df = st.session_state['data']
        
        # EDA options
        eda_option = st.radio("Choose EDA method:", ["Quick Overview", "Custom Visualizations"])
        
        if eda_option == "Quick Overview":
            st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)
            
            # Display basic statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isna().sum().sum())
            col4.metric("Duplicated Rows", df.duplicated().sum())
            
            # Display summary statistics
            st.markdown('<h3>Summary Statistics</h3>', unsafe_allow_html=True)
            st.dataframe(df.describe(include='all').T, use_container_width=True)
            
            # Display missing values
            st.markdown('<h3>Missing Values</h3>', unsafe_allow_html=True)
            missing_data = pd.DataFrame({
                'Column': df.columns,
                'Missing Values': df.isna().sum().values,
                'Percentage': (df.isna().sum() / len(df) * 100).round(2).astype(str) + '%'
            })
            missing_data = missing_data.sort_values('Missing Values', ascending=False)
            
            # Only show columns with missing values
            missing_data = missing_data[missing_data['Missing Values'] > 0]
            
            if len(missing_data) > 0:
                fig = px.bar(
                    missing_data, 
                    x='Column', 
                    y='Missing Values',
                    title='Missing Values by Column',
                    color='Missing Values',
                    text='Percentage'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No missing values found in the dataset.")
            
            # Data types distribution
            st.markdown('<h3>Data Types Distribution</h3>', unsafe_allow_html=True)
            # Convert dtypes to strings to avoid JSON serialization issues
            dtype_counts = pd.DataFrame({
                'Data Type': df.dtypes.astype(str).value_counts().index,
                'Count': df.dtypes.astype(str).value_counts().values
            })
            
            fig = px.pie(
                dtype_counts, 
                values='Count', 
                names='Data Type',
                title='Distribution of Data Types',
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif eda_option == "Custom Visualizations":
            st.markdown('<h2 class="sub-header">Custom Visualizations</h2>', unsafe_allow_html=True)
            
            # Generate custom visualizations
            generate_eda_plots(df)
    
    # Data Cleaning page
    elif choice == "Data Cleaning":
        st.markdown('<h1 class="main-header">Data Cleaning</h1>', unsafe_allow_html=True)
        
        # Check if data is loaded
        if 'data' not in st.session_state:
            st.warning("Please upload your dataset first.")
            return
        
        df = st.session_state['data']
        
        st.markdown("""
        <div class="card">
            <p>This section helps you clean your dataset by handling missing values, removing outliers, and encoding categorical variables.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data cleaning options
        st.markdown('<h2 class="sub-header">Data Cleaning Options</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            missing_threshold = st.slider(
                "Missing Value Threshold (%)",
                min_value=0,
                max_value=100,
                value=50,
                help="Columns with missing values above this threshold will be dropped."
            )
        
        with col2:
            handle_categorical = st.checkbox(
                "One-Hot Encode Categorical Variables",
                value=True,
                help="Convert categorical variables to numerical using one-hot encoding."
            )
        
        # Clean data button
        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                # Clean the data
                df_cleaned, cleaning_report = clean_data(df, threshold_pct=missing_threshold/100, handle_categorical=handle_categorical)
                
                # Store cleaned data in session state
                st.session_state['data_cleaned'] = df_cleaned
                
                # Display cleaning report
                st.markdown('<h2 class="sub-header">Cleaning Report</h2>', unsafe_allow_html=True)
                
                for step in cleaning_report:
                    st.markdown(f"- {step}")
                
                # Compare original and cleaned data
                st.markdown('<h2 class="sub-header">Data Comparison</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<h3>Original Data</h3>', unsafe_allow_html=True)
                    st.metric("Rows", df.shape[0])
                    st.metric("Columns", df.shape[1])
                    st.metric("Missing Values", df.isna().sum().sum())
                
                with col2:
                    st.markdown('<h3>Cleaned Data</h3>', unsafe_allow_html=True)
                    st.metric("Rows", df_cleaned.shape[0])
                    st.metric("Columns", df_cleaned.shape[1])
                    st.metric("Missing Values", df_cleaned.isna().sum().sum())
                
                # Display preview of cleaned data
                st.markdown('<h2 class="sub-header">Cleaned Data Preview</h2>', unsafe_allow_html=True)
                st.dataframe(df_cleaned.head(10))
                
                # Save the cleaned dataframe to a CSV file for later use
                df_cleaned.to_csv('dataset_cleaned.csv', index=None)
                st.success("Data cleaning completed successfully!")
        
        # If cleaned data exists, show it
        elif 'data_cleaned' in st.session_state:
            df_cleaned = st.session_state['data_cleaned']
            
            # Display preview of cleaned data
            st.markdown('<h2 class="sub-header">Cleaned Data Preview</h2>', unsafe_allow_html=True)
            st.dataframe(df_cleaned.head(10))
            
            # Compare original and cleaned data
            st.markdown('<h2 class="sub-header">Data Comparison</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3>Original Data</h3>', unsafe_allow_html=True)
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
                st.metric("Missing Values", df.isna().sum().sum())
            
            with col2:
                st.markdown('<h3>Cleaned Data</h3>', unsafe_allow_html=True)
                st.metric("Rows", df_cleaned.shape[0])
                st.metric("Columns", df_cleaned.shape[1])
                st.metric("Missing Values", df_cleaned.isna().sum().sum())
    
    # Modelling page
    elif choice == "Modelling":
        st.markdown('<h1 class="main-header">Model Training and Evaluation</h1>', unsafe_allow_html=True)
        
        # Check if cleaned data is available
        if 'data_cleaned' in st.session_state:
            df = st.session_state['data_cleaned']
        elif 'data' in st.session_state:
            df = st.session_state['data']
            st.warning("Using original data. It's recommended to clean your data first.")
        else:
            st.warning("Please upload your dataset first.")
            return
        
        st.markdown("""
        <div class="card">
            <p>This section helps you train and evaluate multiple machine learning models on your dataset. Select the target variable and type of problem (classification or regression) to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model training options
        st.markdown('<h2 class="sub-header">Model Training Options</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            problem_type = st.radio("Problem Type", ["Classification", "Regression"])
        
        with col2:
            target_column = st.selectbox("Select Target Column", df.columns)
        
        # Feature selection
        st.markdown('<h3>Feature Selection</h3>', unsafe_allow_html=True)
        
        feature_selection_method = st.radio(
            "Feature Selection Method",
            ["Use all features", "Select specific features"]
        )
        
        selected_features = []
        if feature_selection_method == "Select specific features":
            available_features = [col for col in df.columns if col != target_column]
            selected_features = st.multiselect(
                "Select Features to Include",
                available_features,
                default=available_features
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=25)
                random_state = st.number_input("Random State", min_value=0, max_value=1000, value=42)
            
            with col2:
                # Any additional advanced options can go here
                st.write("Additional options may be added in future versions.")
        
        # Train models button
        if st.button("Train Models"):
            # Prepare data
            if feature_selection_method == "Select specific features" and selected_features:
                X = df[selected_features]
            else:
                X = df.drop([target_column], axis=1)
            
            y = df[target_column]
            
            # Store feature names for later
            feature_names = X.columns.tolist()
            
            with st.spinner("Training models..."):
                # Train and evaluate models
                results, model_objects, X_train, X_test, y_train, y_test = train_and_evaluate_models(
                    X, y, problem_type, test_size=test_size/100, random_state=random_state
                )
                
                # Store results in session state
                st.session_state['model_results'] = results
                st.session_state['model_objects'] = model_objects
                st.session_state['feature_names'] = feature_names
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['problem_type'] = problem_type
                
                # Find best model
                if problem_type == "Classification":
                    best_model = max(results.items(), key=lambda x: x[1]['Accuracy'])
                else:
                    best_model = max(results.items(), key=lambda x: x[1]['R² Score'])
                
                st.session_state['best_model_name'] = best_model[0]
                
                # Save best model
                pickle.dump(model_objects[best_model[0]], open('best_model.pkl', 'wb'))
                
                st.success("Model training completed successfully!")
        
        # If model results exist, show them
        if 'model_results' in st.session_state:
            results = st.session_state['model_results']
            model_objects = st.session_state['model_objects']
            feature_names = st.session_state['feature_names']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            problem_type = st.session_state['problem_type']
            best_model_name = st.session_state['best_model_name']
            
            # Display model comparison
            st.markdown('<h2 class="sub-header">Model Comparison</h2>', unsafe_allow_html=True)
            
            # Convert results to DataFrame for display
            results_df = pd.DataFrame.from_dict(results, orient='index')
            
            # Display results table
            st.dataframe(results_df, use_container_width=True)
            
            # Visualize model comparison
            visualize_model_comparison(results, problem_type)
            
            # Display best model
            st.markdown(f'<h2 class="sub-header">Best Model: {best_model_name}</h2>', unsafe_allow_html=True)
            
            # Display metrics for best model
            col1, col2, col3, col4 = st.columns(4)
            metrics = list(results[best_model_name].items())
            
            for i, (metric, value) in enumerate(metrics):
                cols = [col1, col2, col3, col4]
                cols[i % 4].metric(metric, f"{value:.4f}")
            
            # Feature importance for best model
            st.markdown('<h2 class="sub-header">Feature Importance</h2>', unsafe_allow_html=True)
            visualize_feature_importance(model_objects[best_model_name], feature_names, best_model_name)
            
            # Model performance visualization
            st.markdown('<h2 class="sub-header">Model Performance Visualization</h2>', unsafe_allow_html=True)
            
            if problem_type == "Classification":
                visualize_classification_results(model_objects[best_model_name], X_test, y_test, best_model_name)
            else:
                visualize_regression_results(model_objects[best_model_name], X_test, y_test, best_model_name)
    
    # Download page
    elif choice == "Download":
        st.markdown('<h1 class="main-header">Download Model</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <p>Download your trained model for deployment in your applications. You can also download the cleaned dataset and model evaluation report.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if model exists
        if os.path.exists('best_model.pkl'):
            # Model download
            st.markdown('<h2 class="sub-header">Download Trained Model</h2>', unsafe_allow_html=True)
            
            with open('best_model.pkl', 'rb') as f:
                st.download_button(
                    label="Download Best Model",
                    data=f,
                    file_name="best_model.pkl",
                    mime="application/octet-stream"
                )
            
            if 'best_model_name' in st.session_state:
                st.success(f"Best Model: {st.session_state['best_model_name']}")
                
                # Additional information about the model
                if 'model_results' in st.session_state and 'best_model_name' in st.session_state:
                    best_model_name = st.session_state['best_model_name']
                    results = st.session_state['model_results'][best_model_name]
                    
                    st.markdown('<h3>Model Performance Metrics</h3>', unsafe_allow_html=True)
                    
                    # Display metrics in a more readable format
                    metrics_df = pd.DataFrame({
                        'Metric': list(results.keys()),
                        'Value': list(results.values())
                    })
                    
                    st.dataframe(metrics_df, use_container_width=True)
        else:
            st.warning("No trained model available. Please complete the Modelling section first.")
        
        # Check if cleaned data exists
        if os.path.exists('dataset_cleaned.csv'):
            # Cleaned data download
            st.markdown('<h2 class="sub-header">Download Cleaned Dataset</h2>', unsafe_allow_html=True)
            
            with open('dataset_cleaned.csv', 'rb') as f:
                st.download_button(
                    label="Download Cleaned Dataset",
                    data=f,
                    file_name="dataset_cleaned.csv",
                    mime="text/csv"
                )
        
        # Model report
        if 'model_results' in st.session_state:
            st.markdown('<h2 class="sub-header">Model Evaluation Report</h2>', unsafe_allow_html=True)
            
            # Generate report
            results = st.session_state['model_results']
            problem_type = st.session_state['problem_type']
            
            # Convert results to DataFrame
            results_df = pd.DataFrame.from_dict(results, orient='index')
            
            # Generate CSV report
            csv = results_df.to_csv(index=True)
            st.download_button(
                label="Download Model Evaluation Report",
                data=csv,
                file_name="model_evaluation_report.csv",
                mime="text/csv"
            )
            
            # Display report preview
            st.dataframe(results_df, use_container_width=True)

if __name__ == "__main__":
    main()
