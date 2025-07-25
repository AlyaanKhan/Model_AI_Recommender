import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Model Recommender",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .model-card.best {
        border-left: 4px solid #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    .model-card.second {
        border-left: 4px solid #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    
    .model-card.third {
        border-left: 4px solid #fd7e14;
        background: linear-gradient(135deg, #ffe8d6 0%, #ffd8a8 100%);
    }
</style>
""", unsafe_allow_html=True)

def analyze_data_type(df, target_column):
    """Analyze if the problem is classification or regression"""
    target_data = df[target_column]
    
    # Check if target is categorical
    if target_data.dtype == 'object' or target_data.nunique() <= 10:
        return 'classification'
    else:
        return 'regression'

def get_feature_importance(df, target_column, problem_type):
    """Get feature importance using Random Forest (optimized for speed)"""
    # Sample data if too large
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)
    
    # Handle missing values
    df_clean = df.copy()
    
    # Fill missing values in features
    for col in df_clean.columns:
        if col != target_column:
            if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
                # Handle categorical columns properly
                if df_clean[col].dtype.name == 'category':
                    # Add 'Unknown' to categories if it doesn't exist
                    if 'Unknown' not in df_clean[col].cat.categories:
                        df_clean[col] = df_clean[col].cat.add_categories(['Unknown'])
                    df_clean[col] = df_clean[col].fillna('Unknown')
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Fill missing values in target
    if df_clean[target_column].dtype == 'object' or df_clean[target_column].dtype.name == 'category':
        # Handle categorical columns properly
        if df_clean[target_column].dtype.name == 'category':
            # Add 'Unknown' to categories if it doesn't exist
            if 'Unknown' not in df_clean[target_column].cat.categories:
                df_clean[target_column] = df_clean[target_column].cat.add_categories(['Unknown'])
            df_clean[target_column] = df_clean[target_column].fillna('Unknown')
        else:
            df_clean[target_column] = df_clean[target_column].fillna('Unknown')
    else:
        df_clean[target_column] = df_clean[target_column].fillna(df_clean[target_column].median())
    
    # Remove rows where target is still missing (if any)
    df_clean = df_clean.dropna(subset=[target_column])
    
    if len(df_clean) == 0:
        return pd.DataFrame({'feature': [], 'importance': []})
    
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # Handle categorical features
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
    
    if problem_type == 'classification':
        model = RandomForestClassifier(n_estimators=25, random_state=42)  # Reduced for speed
    else:
        model = RandomForestRegressor(n_estimators=25, random_state=42)  # Reduced for speed
    
    model.fit(X, y)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance

def evaluate_models(df, target_column, problem_type):
    """Evaluate different models and return rankings"""
    # Sample data if too large for faster processing
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
        st.info("📊 Large dataset detected! Using 10,000 random samples for faster analysis.")
    
    # Handle missing values
    df_clean = df.copy()
    
    # Count missing values
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        st.info(f"🔧 Handling {missing_count} missing values automatically...")
    
    # Fill missing values in features
    for col in df_clean.columns:
        if col != target_column:
            if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
                # Handle categorical columns properly
                if df_clean[col].dtype.name == 'category':
                    # Add 'Unknown' to categories if it doesn't exist
                    if 'Unknown' not in df_clean[col].cat.categories:
                        df_clean[col] = df_clean[col].cat.add_categories(['Unknown'])
                    df_clean[col] = df_clean[col].fillna('Unknown')
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Fill missing values in target
    if df_clean[target_column].dtype == 'object' or df_clean[target_column].dtype.name == 'category':
        # Handle categorical columns properly
        if df_clean[target_column].dtype.name == 'category':
            # Add 'Unknown' to categories if it doesn't exist
            if 'Unknown' not in df_clean[target_column].cat.categories:
                df_clean[target_column] = df_clean[target_column].cat.add_categories(['Unknown'])
            df_clean[target_column] = df_clean[target_column].fillna('Unknown')
        else:
            df_clean[target_column] = df_clean[target_column].fillna('Unknown')
    else:
        df_clean[target_column] = df_clean[target_column].fillna(df_clean[target_column].median())
    
    # Remove rows where target is still missing (if any)
    df_clean = df_clean.dropna(subset=[target_column])
    
    if len(df_clean) == 0:
        st.error("❌ No valid data remaining after handling missing values. Please check your dataset.")
        return [], {}
    
    # Show data cleaning summary
    if len(df_clean) < len(df):
        st.success(f"✅ Data cleaned: {len(df_clean)} rows remaining (removed {len(df) - len(df_clean)} rows with missing target values)")
    
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # Handle categorical features
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle categorical target for classification
    if problem_type == 'classification':
        y = le.fit_transform(y.astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    scores = {}
    
    if problem_type == 'classification':
        # Classification models (optimized for speed)
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        
        for name, model in models.items():
            if name in ['Logistic Regression', 'SVM']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            scores[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
    
    else:
        # Regression models (optimized for speed)
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
        }
        
        for name, model in models.items():
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            scores[name] = {
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
    
    # Rank models
    if problem_type == 'classification':
        # Rank by F1 score
        rankings = sorted(scores.items(), key=lambda x: x[1]['f1'], reverse=True)
    else:
        # Rank by R² score
        rankings = sorted(scores.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    return rankings, scores

def suggest_best_target(df):
    """Analyze data and suggest the best target column"""
    
    # Define common target column patterns
    target_patterns = {
        'classification': [
            'target', 'label', 'class', 'category', 'type', 'status', 'outcome',
            'result', 'prediction', 'decision', 'churn', 'fraud', 'spam',
            'buy', 'click', 'convert', 'engage', 'subscribe', 'purchase',
            'success', 'failure', 'yes_no', 'true_false', 'approved', 'rejected'
        ],
        'regression': [
            'price', 'cost', 'value', 'amount', 'revenue', 'sales', 'profit',
            'score', 'rating', 'rate', 'percentage', 'ratio', 'index',
            'temperature', 'pressure', 'weight', 'height', 'age', 'duration',
            'count', 'number', 'quantity', 'volume', 'size', 'length'
        ]
    }
    
    # Check if dataset has no clear target (all columns look like features)
    feature_indicators = ['id', 'index', 'name', 'description', 'text', 'date', 'time', 'timestamp', 'customer', 'product', 'location', 'category', 'frequency', 'amount', 'income', 'age', 'followers', 'opens', 'visits', 'purchase']
    all_feature_like = True
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if column looks like a target
        is_target_like = any(pattern in col_lower for patterns in target_patterns.values() for pattern in patterns)
        is_feature_like = any(indicator in col_lower for indicator in feature_indicators)
        
        if is_target_like and not is_feature_like:
            all_feature_like = False
            break
    
    best_target = None
    best_score = 0
    best_reason = ""
    
    for col in df.columns:
        score = 0
        reasons = []
        
        # Check for missing values (fewer is better)
        missing_pct = df[col].isnull().sum() / len(df)
        if missing_pct < 0.1:  # Less than 10% missing
            score += 20
            reasons.append("low missing values")
        elif missing_pct < 0.3:  # Less than 30% missing
            score += 10
            reasons.append("moderate missing values")
        
        # Check for pattern matching
        col_lower = col.lower()
        matched_pattern = None
        for pattern_type, patterns in target_patterns.items():
            for pattern in patterns:
                if pattern in col_lower:
                    score += 15
                    matched_pattern = pattern
                    break
            if matched_pattern:
                break
        
        if matched_pattern:
            reasons.append(f"matches '{matched_pattern}' pattern")
        
        # Check data type and distribution
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 10:
                score += 25
                reasons.append(f"categorical with {unique_count} classes (good for classification)")
            elif unique_count > 10:
                score += 5
                reasons.append(f"many categories ({unique_count})")
        else:  # numeric
            unique_count = df[col].nunique()
            if unique_count <= 10:
                score += 20
                reasons.append(f"numeric with few values ({unique_count}) - good for classification")
            else:
                score += 15
                reasons.append(f"continuous numeric ({unique_count} unique values) - good for regression")
        
        # Check for balanced distribution (for classification)
        if df[col].dtype == 'object' or df[col].nunique() <= 10:
            value_counts = df[col].value_counts()
            if len(value_counts) >= 2:
                min_count = value_counts.min()
                max_count = value_counts.max()
                balance_ratio = min_count / max_count
                if balance_ratio > 0.3:
                    score += 10
                    reasons.append("well-balanced classes")
                elif balance_ratio > 0.1:
                    score += 5
                    reasons.append("moderately balanced classes")
        
        # Check for business logic indicators
        business_indicators = {
            'classification': ['target', 'label', 'class', 'status', 'outcome', 'result', 'prediction'],
            'regression': ['price', 'cost', 'value', 'score', 'rating', 'satisfaction', 'performance']
        }
        
        # Penalize clearly feature-like columns
        feature_penalties = ['customer', 'product', 'location', 'category', 'frequency', 'amount', 'income', 'age', 'followers', 'opens', 'visits', 'purchase', 'date', 'time']
        is_feature_like = any(penalty in col_lower for penalty in feature_penalties)
        
        if is_feature_like:
            score -= 30  # Heavy penalty for feature-like columns
            reasons.append("appears to be a feature column")
        
        for indicator_type, indicators in business_indicators.items():
            for indicator in indicators:
                if indicator in col_lower:
                    score += 20
                    reasons.append(f"business target indicator ({indicator_type})")
                    break
        
        # Update best target if this column has higher score
        if score > best_score:
            best_score = score
            best_target = col
            best_reason = ", ".join(reasons)
    
    # If no good target found or all columns look like features, return None
    if best_score < 50 or all_feature_like:
        return None, "No clear target column detected - all columns appear to be features"
    
    return best_target, best_reason

def calculate_confidence(df, target_column):
    """Calculate confidence score for target column suggestion"""
    
    confidence = 0
    reasons = []
    
    # Check missing values
    missing_pct = df[target_column].isnull().sum() / len(df)
    if missing_pct < 0.05:
        confidence += 25
        reasons.append("very low missing values")
    elif missing_pct < 0.1:
        confidence += 20
        reasons.append("low missing values")
    elif missing_pct < 0.2:
        confidence += 15
        reasons.append("moderate missing values")
    
    # Check data distribution
    if df[target_column].dtype == 'object':
        unique_count = df[target_column].nunique()
        if 2 <= unique_count <= 5:
            confidence += 25
            reasons.append("ideal categorical distribution")
        elif 6 <= unique_count <= 10:
            confidence += 20
            reasons.append("good categorical distribution")
    else:
        unique_count = df[target_column].nunique()
        if unique_count > 50:
            confidence += 25
            reasons.append("good continuous distribution")
        elif 10 < unique_count <= 50:
            confidence += 20
            reasons.append("moderate continuous distribution")
    
    # Check for target-like naming
    target_keywords = ['target', 'label', 'class', 'outcome', 'result', 'prediction']
    if any(keyword in target_column.lower() for keyword in target_keywords):
        confidence += 20
        reasons.append("target-like naming")
    
    # Determine confidence level
    if confidence >= 70:
        confidence_text = "High confidence - Excellent choice"
    elif confidence >= 50:
        confidence_text = "Medium confidence - Good choice"
    elif confidence >= 30:
        confidence_text = "Low confidence - Consider alternatives"
    else:
        confidence_text = "Very low confidence - Manual review recommended"
    
    return confidence, confidence_text

def get_alternative_targets(df, best_target):
    """Get alternative target column suggestions"""
    
    alternatives = []
    
    for col in df.columns:
        if col == best_target:
            continue
            
        score = 0
        reason = ""
        
        # Check for missing values
        missing_pct = df[col].isnull().sum() / len(df)
        if missing_pct < 0.1:
            score += 10
            reason += "low missing values, "
        
        # Check data type and distribution
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 10:
                score += 15
                reason += f"categorical with {unique_count} classes, "
        else:
            unique_count = df[col].nunique()
            if unique_count > 10:
                score += 15
                reason += f"continuous numeric ({unique_count} values), "
        
        # Check for target-like naming
        target_keywords = ['target', 'label', 'class', 'outcome', 'result', 'prediction', 'price', 'cost', 'value']
        if any(keyword in col.lower() for keyword in target_keywords):
            score += 10
            reason += "target-like naming, "
        
        if score > 20 and reason:
            reason = reason.rstrip(", ")
            alternatives.append((col, reason))
    
    # Sort by score and return top alternatives
    alternatives.sort(key=lambda x: len(x[1]), reverse=True)  # Sort by reason length as proxy for score
    return alternatives

def generate_model_code(df, target_column, problem_type, model_name, filename):
    """Generate complete Python code for the best model"""
    
    # Get column types for preprocessing
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from feature lists
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    # Model imports and setup
    if problem_type == 'classification':
        if model_name == 'Random Forest':
            model_import = "from sklearn.ensemble import RandomForestClassifier"
            model_init = "model = RandomForestClassifier(n_estimators=100, random_state=42)"
        elif model_name == 'Gradient Boosting':
            model_import = "from sklearn.ensemble import GradientBoostingClassifier"
            model_init = "model = GradientBoostingClassifier(n_estimators=100, random_state=42)"
        elif model_name == 'Logistic Regression':
            model_import = "from sklearn.linear_model import LogisticRegression"
            model_init = "model = LogisticRegression(random_state=42, max_iter=1000)"
        elif model_name == 'K-Nearest Neighbors':
            model_import = "from sklearn.neighbors import KNeighborsClassifier"
            model_init = "model = KNeighborsClassifier(n_neighbors=5)"
        elif model_name == 'Naive Bayes':
            model_import = "from sklearn.naive_bayes import GaussianNB"
            model_init = "model = GaussianNB()"
        else:
            model_import = "from sklearn.ensemble import RandomForestClassifier"
            model_init = "model = RandomForestClassifier(n_estimators=100, random_state=42)"
    else:  # regression
        if model_name == 'Random Forest':
            model_import = "from sklearn.ensemble import RandomForestRegressor"
            model_init = "model = RandomForestRegressor(n_estimators=100, random_state=42)"
        elif model_name == 'Gradient Boosting':
            model_import = "from sklearn.ensemble import GradientBoostingRegressor"
            model_init = "model = GradientBoostingRegressor(n_estimators=100, random_state=42)"
        elif model_name == 'Linear Regression':
            model_import = "from sklearn.linear_model import LinearRegression"
            model_init = "model = LinearRegression()"
        elif model_name == 'Ridge Regression':
            model_import = "from sklearn.linear_model import Ridge"
            model_init = "model = Ridge(random_state=42)"
        elif model_name == 'K-Nearest Neighbors':
            model_import = "from sklearn.neighbors import KNeighborsRegressor"
            model_init = "model = KNeighborsRegressor(n_neighbors=5)"
        else:
            model_import = "from sklearn.ensemble import RandomForestRegressor"
            model_init = "model = RandomForestRegressor(n_estimators=100, random_state=42)"
    
    # Generate the complete code
    code = f'''# {model_name} Model Implementation
# Generated by AI Model Recommender
# Problem Type: {problem_type.title()}
# Target Column: {target_column}
# Data File: {filename}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
{model_import}
from sklearn.metrics import {'accuracy_score, precision_score, recall_score, f1_score' if problem_type == 'classification' else 'mean_squared_error, r2_score, mean_absolute_error'}

# Load your data
df = pd.read_csv('{filename}')

# Separate features and target
X = df.drop(columns=['{target_column}'])
y = df['{target_column}']

# Handle categorical features
le = LabelEncoder()
categorical_columns = {categorical_cols}
for col in categorical_columns:
    if col in X.columns:
        X[col] = le.fit_transform(X[col].astype(str))

# Handle categorical target for classification
if '{problem_type}' == 'classification':
    y = le.fit_transform(y.astype(str))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
{model_init}

# Train the model (use scaled data for models that need it)
if '{model_name}' in ['Logistic Regression', 'Linear Regression', 'Ridge Regression']:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
else:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Evaluate the model
print("\\n=== Model Performance ===")
'''
    
    if problem_type == 'classification':
        code += f'''
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {{accuracy:.3f}}")
print(f"Precision: {{precision:.3f}}")
print(f"Recall: {{recall:.3f}}")
print(f"F1 Score: {{f1:.3f}}")
'''
    else:
        code += f'''
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {{mse:.3f}}")
print(f"Root Mean Squared Error: {{rmse:.3f}}")
print(f"Mean Absolute Error: {{mae:.3f}}")
print(f"R² Score: {{r2:.3f}}")
'''
    
    code += f'''

# Feature importance (for tree-based models)
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({{
        'feature': X.columns,
        'importance': model.feature_importances_
    }}).sort_values('importance', ascending=False)
    
    print("\\n=== Top 10 Most Important Features ===")
    print(feature_importance.head(10))

# Save the model (optional)
import joblib
joblib.dump(model, '{model_name.lower().replace(" ", "_")}_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\\nModel saved as '{model_name.lower().replace(" ", "_")}_model.pkl'")

# Make predictions on new data (example)
print("\\n=== Making Predictions ===")
print("To make predictions on new data, use:")
print("new_data = pd.read_csv('new_data.csv')")
print("new_data_scaled = scaler.transform(new_data)")
print("predictions = model.predict(new_data_scaled)")
'''
    
    return code

def create_visualizations(df, target_column, problem_type):
    """Create comprehensive visualizations (optimized for speed)"""
    # Sample data for faster visualization if too large
    if len(df) > 5000:
        df_viz = df.sample(n=5000, random_state=42)
    else:
        df_viz = df
    
    # Handle missing values for visualization
    df_viz = df_viz.copy()
    
    # Fill missing values in target column for visualization
    if df_viz[target_column].dtype == 'object' or df_viz[target_column].dtype.name == 'category':
        # Handle categorical columns properly
        if df_viz[target_column].dtype.name == 'category':
            # Add 'Unknown' to categories if it doesn't exist
            if 'Unknown' not in df_viz[target_column].cat.categories:
                df_viz[target_column] = df_viz[target_column].cat.add_categories(['Unknown'])
            df_viz[target_column] = df_viz[target_column].fillna('Unknown')
        else:
            df_viz[target_column] = df_viz[target_column].fillna('Unknown')
    else:
        df_viz[target_column] = df_viz[target_column].fillna(df_viz[target_column].median())
    
    # Remove rows where target is still missing
    df_viz = df_viz.dropna(subset=[target_column])
    
    # Create simpler, faster visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Target Distribution")
        if problem_type == 'classification':
            target_counts = df_viz[target_column].value_counts().head(10)  # Limit to top 10
            fig1 = px.bar(x=target_counts.index, y=target_counts.values, 
                         title="Target Variable Distribution")
        else:
            fig1 = px.histogram(df_viz, x=target_column, title="Target Variable Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Feature Importance")
        # Simplified feature importance (top 10 features only)
        importance = get_feature_importance(df_viz, target_column, problem_type).head(10)
        fig2 = px.bar(importance, x='feature', y='importance', 
                     title="Top 10 Most Important Features")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Correlation heatmap (only for numeric columns, limited size)
    st.markdown("### 🔗 Correlation Heatmap")
    numeric_df = df_viz.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1 and len(numeric_df.columns) <= 20:  # Limit to 20 columns
        corr_matrix = numeric_df.corr()
        fig3 = px.imshow(corr_matrix, title="Feature Correlations")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("📊 Correlation heatmap skipped (too many columns or no numeric data)")
    
    # Data overview as metrics instead of table
    st.markdown("### 📋 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Problem Type", problem_type.title())

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Model Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Upload your CSV file and let AI recommend the best models for your data!</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📊 Upload Your Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Display basic info
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Target column selection with AI suggestions
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('<h3>🎯 Select Target Column</h3>', unsafe_allow_html=True)
            
            # Get AI suggestions for best target column
            suggested_target, suggestion_reason = suggest_best_target(df)
            
            if suggested_target:
                # Calculate confidence score
                confidence_score, confidence_text = calculate_confidence(df, suggested_target)
                
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f'<h4>🤖 AI Suggestion: <strong>{suggested_target}</strong></h4>', unsafe_allow_html=True)
                st.markdown(f'<p><strong>Confidence:</strong> {confidence_score}% - {confidence_text}</p>', unsafe_allow_html=True)
                st.markdown(f'<p><strong>Why this column?</strong> {suggestion_reason}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show alternative suggestions
                alternative_targets = get_alternative_targets(df, suggested_target)
                if alternative_targets:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown('<h4>🔄 Alternative Suggestions</h4>', unsafe_allow_html=True)
                    for alt_target, alt_reason in alternative_targets[:3]:  # Show top 3 alternatives
                        st.markdown(f'• **{alt_target}**: {alt_reason}')
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # No suitable target column found
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown('<h4>⚠️ No Clear Target Column Found</h4>', unsafe_allow_html=True)
                st.markdown("""
                **Your dataset doesn't contain an obvious target column for prediction. Here are your options:**
                
                1. **Create a target column** from existing features
                2. **Use clustering** to find patterns in your data
                3. **Select any column** as your target
                4. **Upload a different dataset** with a target column
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show target creation options
                st.markdown('<h4>🔧 Create Target Column</h4>', unsafe_allow_html=True)
                target_creation_option = st.selectbox(
                    "Choose how to create a target:",
                    ["Select existing column", "Create binary target", "Create categorical target", "Create numeric target", "Use clustering"],
                    help="Options to create a target variable from your data"
                )
                
                if target_creation_option == "Create binary target":
                    base_column = st.selectbox("Select base column:", df.select_dtypes(include=[np.number]).columns.tolist())
                    threshold = st.number_input("Threshold value:", value=df[base_column].median())
                    operation = st.selectbox("Operation:", ["Greater than", "Less than", "Equal to"])
                    
                    if st.button("Create Binary Target"):
                        if operation == "Greater than":
                            df['target_binary'] = (df[base_column] > threshold).astype(int)
                        elif operation == "Less than":
                            df['target_binary'] = (df[base_column] < threshold).astype(int)
                        else:
                            df['target_binary'] = (df[base_column] == threshold).astype(int)
                        
                        st.success(f"✅ Created binary target: {operation} {threshold}")
                        suggested_target = 'target_binary'
                
                elif target_creation_option == "Create categorical target":
                    base_column = st.selectbox("Select base column:", df.select_dtypes(include=[np.number]).columns.tolist())
                    num_bins = st.slider("Number of categories:", 2, 5, 3)
                    
                    if st.button("Create Categorical Target"):
                        df['target_categorical'] = pd.cut(df[base_column], bins=num_bins, labels=[f'Category_{i+1}' for i in range(num_bins)])
                        st.success(f"✅ Created categorical target with {num_bins} categories")
                        suggested_target = 'target_categorical'
                
                elif target_creation_option == "Create numeric target":
                    col1, col2 = st.columns(2)
                    with col1:
                        col1_name = st.selectbox("Select first column:", df.select_dtypes(include=[np.number]).columns.tolist())
                    with col2:
                        operation = st.selectbox("Operation:", ["Add", "Subtract", "Multiply", "Divide"])
                    
                    if len(df.select_dtypes(include=[np.number]).columns) > 1:
                        col2_name = st.selectbox("Select second column:", [col for col in df.select_dtypes(include=[np.number]).columns if col != col1_name])
                        
                        if st.button("Create Numeric Target"):
                            if operation == "Add":
                                df['target_numeric'] = df[col1_name] + df[col2_name]
                            elif operation == "Subtract":
                                df['target_numeric'] = df[col1_name] - df[col2_name]
                            elif operation == "Multiply":
                                df['target_numeric'] = df[col1_name] * df[col2_name]
                            else:  # Divide
                                df['target_numeric'] = df[col1_name] / df[col2_name].replace(0, 1)  # Avoid division by zero
                            
                            st.success(f"✅ Created numeric target: {col1_name} {operation.lower()} {col2_name}")
                            suggested_target = 'target_numeric'
                
                elif target_creation_option == "Use clustering":
                    st.markdown("""
                    **🔍 Clustering Analysis**
                    
                    Clustering will group your data into similar patterns without needing a target column.
                    This is useful for:
                    - Customer segmentation
                    - Market analysis
                    - Pattern discovery
                    - Anomaly detection
                    """)
                    
                    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
                    
                    if st.button("Perform Clustering"):
                        # Perform K-means clustering
                        from sklearn.cluster import KMeans
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.impute import SimpleImputer
                        
                        # Prepare data for clustering
                        numeric_data = df.select_dtypes(include=[np.number])
                        if len(numeric_data.columns) > 0:
                            # Handle missing values in numeric data
                            imputer = SimpleImputer(strategy='median')
                            numeric_data_clean = pd.DataFrame(
                                imputer.fit_transform(numeric_data),
                                columns=numeric_data.columns,
                                index=numeric_data.index
                            )
                            
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(numeric_data_clean)
                            
                            # Perform clustering
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            clusters = kmeans.fit_predict(scaled_data)
                            
                            # Add cluster labels to dataframe
                            df['cluster_target'] = clusters
                            
                            st.success(f"✅ Created {n_clusters} clusters using K-means")
                            suggested_target = 'cluster_target'
                            
                            # Show cluster analysis
                            st.markdown("### 📊 Cluster Analysis")
                            cluster_counts = df['cluster_target'].value_counts().sort_index()
                            st.bar_chart(cluster_counts)
                        else:
                            st.error("❌ No numeric columns found for clustering")
            
            target_column = st.selectbox(
                "Choose the column you want to predict:",
                df.columns,
                index=df.columns.get_loc(suggested_target) if suggested_target in df.columns else 0,
                help="This column will be your target variable (what you want to predict)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if target_column:
                # Analyze problem type
                problem_type = analyze_data_type(df, target_column)
                
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f'<h3>✅ Problem Type Detected: <strong>{problem_type.upper()}</strong></h3>', unsafe_allow_html=True)
                
                if problem_type == 'classification':
                    st.markdown("""
                    **Classification Problem Detected** because:
                    - Your target column contains categorical values
                    - Or has a limited number of unique values (≤10)
                    """)
                else:
                    st.markdown("""
                    **Regression Problem Detected** because:
                    - Your target column contains continuous numerical values
                    - And has many unique values (>10)
                    """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Evaluate models
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🤖 Starting model analysis...")
                progress_bar.progress(10)
                
                rankings, scores = evaluate_models(df, target_column, problem_type)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                st.success("Model analysis completed successfully!")
                
                # Display model recommendations
                st.markdown('<h2 class="sub-header">🏆 Model Recommendations</h2>', unsafe_allow_html=True)
                
                for i, (model_name, metrics) in enumerate(rankings[:3]):
                    if i == 0:
                        card_class = "model-card best"
                        rank_emoji = "🥇"
                        rank_text = "BEST CHOICE"
                    elif i == 1:
                        card_class = "model-card second"
                        rank_emoji = "🥈"
                        rank_text = "SECOND BEST"
                    else:
                        card_class = "model-card third"
                        rank_emoji = "🥉"
                        rank_text = "THIRD BEST"
                    
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    st.markdown(f'<h3>{rank_emoji} {model_name} - {rank_text}</h3>', unsafe_allow_html=True)
                    
                    if problem_type == 'classification':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        with col2:
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                        with col3:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                        with col4:
                            st.metric("F1 Score", f"{metrics['f1']:.3f}")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{metrics['r2']:.3f}")
                        with col2:
                            st.metric("MSE", f"{metrics['mse']:.3f}")
                        with col3:
                            st.metric("RMSE", f"{metrics['rmse']:.3f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Model explanations
                st.markdown('<h2 class="sub-header">💡 Why These Models?</h2>', unsafe_allow_html=True)
                
                model_explanations = {
                    'Random Forest': 'Excellent for both classification and regression. Handles non-linear relationships, feature interactions, and provides feature importance.',
                    'Gradient Boosting': 'High performance model that builds strong learners from weak ones. Great for complex patterns in data.',
                    'Logistic Regression': 'Simple, interpretable, and fast. Good baseline for classification problems.',
                    'Linear Regression': 'Simple and interpretable. Good baseline for regression problems.',
                    'Ridge Regression': 'Linear regression with regularization to prevent overfitting.',
                    'Lasso Regression': 'Linear regression with feature selection through L1 regularization.',
                    'SVM': 'Good for high-dimensional data and non-linear relationships.',
                    'K-Nearest Neighbors': 'Simple and effective for small to medium datasets.',
                    'Naive Bayes': 'Fast and works well with categorical features.'
                }
                
                for model_name, _ in rankings[:3]:
                    st.markdown(f'**{model_name}**: {model_explanations.get(model_name, "A powerful machine learning model.")}')
                
                # Generate ready-to-use code for the best model
                st.markdown('<h2 class="sub-header">💻 Ready-to-Use Code</h2>', unsafe_allow_html=True)
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("""
                **🚀 Complete Implementation Code**
                
                Below is the complete Python code for the best model. Just copy, paste, and run with your data!
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                best_model_name, best_model_metrics = rankings[0]
                generated_code = generate_model_code(df, target_column, problem_type, best_model_name, uploaded_file.name)
                
                # Display code with syntax highlighting
                st.code(generated_code, language='python')
                
                # Download button for the code
                st.download_button(
                    label="📥 Download Code as .py file",
                    data=generated_code,
                    file_name=f"{best_model_name.lower().replace(' ', '_')}_model.py",
                    mime="text/plain"
                )
                
                # Usage instructions
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("""
                **🎯 How to Use This Code:**
                
                1. **Copy the code above** or download the .py file
                2. **Make sure your CSV file '{uploaded_file.name}' is in the same folder**
                3. **Run the script** - it will automatically:
                   - Load and preprocess your data
                   - Train the best model
                   - Show performance metrics
                   - Save the trained model
                   - Show how to make predictions
                
                **📦 Required Packages:** `pandas`, `numpy`, `scikit-learn`, `joblib`
                
                **💻 Install with:** `pip install pandas numpy scikit-learn joblib`
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualizations
                st.markdown('<h2 class="sub-header">📈 Data Analysis</h2>', unsafe_allow_html=True)
                with st.spinner('📊 Creating visualizations...'):
                    create_visualizations(df, target_column, problem_type)
                
                # Data preview
                st.markdown('<h2 class="sub-header">👀 Data Preview</h2>', unsafe_allow_html=True)
                st.dataframe(df.head(5), use_container_width=True)  # Reduced to 5 rows for speed
                
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("""
            **Troubleshooting Tips:**
            - Make sure your CSV file is properly formatted
            - Check that the file contains numerical or categorical data
            - Ensure there are no encoding issues
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Welcome message
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("""
        ## 🚀 Welcome to AI Model Recommender!
        
        **How it works:**
        1. 📁 Upload your CSV file using the sidebar
        2. 🎯 Select the column you want to predict
        3. 🤖 Our AI will automatically detect if it's a classification or regression problem
        4. 🏆 Get ranked recommendations for the best models
        5. 📊 View comprehensive data analysis and visualizations
        
        **Supported file formats:** CSV
        
        **What you'll get:**
        - ✅ Automatic problem type detection (Classification vs Regression)
        - 🏆 Top 3 model recommendations with performance metrics
        - 📈 Data visualizations and analysis
        - 💡 Explanations for why each model is recommended
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Example section
        st.markdown('<h2 class="sub-header">📚 Example Use Cases</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Classification Examples:**
            - Customer churn prediction
            - Spam email detection
            - Disease diagnosis
            - Credit card fraud detection
            """)
        
        with col2:
            st.markdown("""
            **Regression Examples:**
            - House price prediction
            - Sales forecasting
            - Temperature prediction
            - Stock price prediction
            """)

if __name__ == "__main__":
    main() 