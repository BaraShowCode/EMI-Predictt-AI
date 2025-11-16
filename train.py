import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Suppress warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# --- 1. Data Loading and Preprocessing (Step 1) ---
def load_and_clean_data(filepath):
    """Loads and performs initial cleaning of the dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {df.shape}")

    # Drop rows where targets are missing (critical for training)
    df.dropna(subset=['emi_eligibility', 'max_monthly_emi'], inplace=True)
    
    # Handle duplicates
    df.drop_duplicates(inplace=True)
    
    # Convert ALL numeric columns to numeric *before* imputation
    num_cols_to_convert = [
        'age', 'monthly_salary', 'credit_score', 'current_emi_amount', 
        'bank_balance', 'emergency_fund', 'monthly_rent', 'school_fees', 
        'college_fees', 'travel_expenses', 'groceries_utilities', 
        'other_monthly_expenses', 'requested_amount', 'requested_tenure',
        'family_size', 'dependents'
    ]
    for col in num_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Impute missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    df['monthly_salary'].fillna(df['monthly_salary'].median(), inplace=True)
    df['credit_score'].fillna(df['credit_score'].median(), inplace=True)
    df['current_emi_amount'].fillna(0, inplace=True)
    df['bank_balance'].fillna(df['bank_balance'].median(), inplace=True)
    df['emergency_fund'].fillna(0, inplace=True)
    
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
    for col in expense_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    print(f"Data cleaned. Shape after cleaning: {df.shape}")
    return df

# --- 2. Exploratory Data Analysis (Step 2) ---
def perform_eda(df, output_dir="eda_charts"):
    """Generates and saves EDA charts."""
    print("Starting Exploratory Data Analysis (EDA)...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. EMI Eligibility Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='emi_eligibility', palette='viridis')
    plt.title('Distribution of EMI Eligibility (Target)')
    plt.savefig(f"{output_dir}/1_eligibility_distribution.png")
    plt.close()

    # 2. Max EMI Distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(df['max_monthly_emi'], kde=True, bins=50, color='blue')
    plt.title('Distribution of Max Monthly EMI (Target)')
    plt.savefig(f"{output_dir}/2_max_emi_distribution.png")
    plt.close()

    # 3. Eligibility by Scenario
    plt.figure(figsize=(14, 7))
    sns.countplot(data=df, x='emi_scenario', hue='emi_eligibility', palette='coolwarm')
    plt.title('EMI Eligibility by Lending Scenario')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_eligibility_by_scenario.png")
    plt.close()

    # 4. Correlation Matrix
    financial_vars = ['monthly_salary', 'credit_score', 'bank_balance', 'current_emi_amount', 'requested_amount', 'max_monthly_emi']
    df[financial_vars] = df[financial_vars].apply(pd.to_numeric, errors='coerce').fillna(0)
    corr_matrix = df[financial_vars].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu', fmt='.2f')
    plt.title('Correlation Matrix of Key Financial Variables')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_correlation_heatmap.png")
    plt.close()

    # 5. Credit Score by Education
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='education', y='credit_score', palette='muted')
    plt.title('Credit Score Distribution by Education Level')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/5_credit_score_by_education.png")
    plt.close()
    
    print(f"EDA charts saved to '{output_dir}' folder.")

# --- 3. Feature Engineering (Step 3) ---
def feature_engineer(data):
    """Creates derived financial ratios and risk scoring features."""
    print("Applying feature engineering...")
    df_feat = data.copy()
    
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
    df_feat['total_expenses'] = df_feat[expense_cols].sum(axis=1)
    
    df_feat['total_debt'] = df_feat['total_expenses'] + df_feat['current_emi_amount']
    df_feat['dti_ratio'] = df_feat['total_debt'] / (df_feat['monthly_salary'] + 1e-6)
    
    df_feat['total_savings'] = df_feat['bank_balance'] + df_feat['emergency_fund']
    df_feat['savings_to_income_ratio'] = df_feat['total_savings'] / (df_feat['monthly_salary'] + 1e-6)
    
    df_feat['disposable_income'] = df_feat['monthly_salary'] - df_feat['total_debt']
    df_feat['loan_affordability_ratio'] = df_feat['requested_amount'] / (df_feat['disposable_income'] + 1e-6)
    
    df_feat['financial_risk_score'] = (df_feat['credit_score'] / 850) + (df_feat['savings_to_income_ratio'] * 0.5) - (df_feat['dti_ratio'] * 0.5)
    
    df_feat['age_x_salary'] = df_feat['age'] * df_feat['monthly_salary']
    df_feat['credit_score_x_dti'] = df_feat['credit_score'] * df_feat['dti_ratio']
    
    df_feat.replace([np.inf, -np.inf], 0, inplace=True)
    df_feat.fillna(0, inplace=True)
    
    print("Feature engineering complete.")
    return df_feat

# --- 4. Model Training & MLflow (Step 4 & 5) ---

def get_preprocessor(X):
    """Identifies feature types and creates a preprocessor pipeline."""
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='drop')
    
    return preprocessor

def train_classification(X_train, X_val, y_train, y_val, preprocessor):
    """Trains, evaluates, and logs 3 classification models."""
    print("\n--- Starting Classification Model Training... ---")
    mlflow.set_experiment("EMI_Eligibility_Classification_v2")
    
    label_map = {'Eligible': 0, 'High_Risk': 1, 'Not_Eligible': 2}
    y_train_mapped = y_train.map(label_map)
    y_val_mapped = y_val.map(label_map)
    
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', objective='multi:softmax', num_class=3)
    }
    
    best_f1 = -1
    best_model_name = ""
    best_model_pipe = None
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"  Training {name}...")
            if name == "XGBoost":
                model.fit(X_train_processed, y_train_mapped)
                y_pred_mapped = model.predict(X_val_processed)
                reverse_map = {v: k for k, v in label_map.items()}
                y_pred = pd.Series(y_pred_mapped).map(reverse_map)
                y_proba = model.predict_proba(X_val_processed)
            else:
                model.fit(X_train_processed, y_train)
                y_pred = model.predict(X_val_processed)
                y_proba = model.predict_proba(X_val_processed)
            
            pipe_to_log = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_val, y_proba, multi_class='ovr')
            
            mlflow.log_params(model.get_params())
            mlflow.log_metric("val_accuracy", acc)
            mlflow.log_metric("val_f1_score", f1)
            mlflow.log_metric("val_precision", precision)
            mlflow.log_metric("val_recall", recall)
            mlflow.log_metric("val_roc_auc_ovr", roc_auc)
            
            mlflow.sklearn.log_model(pipe_to_log, f"{name}_model")
            print(f"  {name} model trained. Validation F1-Score: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_pipe = pipe_to_log
                best_model_name = name
    
    print(f"--- Best Classification Model: {best_model_name} (F1: {best_f1:.4f}) ---")
    
    print(f"Registering '{best_model_name}' as 'production'...")
    mlflow.sklearn.log_model(
        best_model_pipe,
        "classification_model",
        registered_model_name="EMIPredict_Classification_Model"
    )
    return best_model_pipe

def train_regression(X_train, X_val, y_train, y_val, preprocessor):
    """Trains, evaluates, and logs 3 regression models."""
    print("\n--- Starting Regression Model Training... ---")
    mlflow.set_experiment("EMI_Amount_Regression_v2")
    
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "XGBoostRegressor": XGBRegressor(random_state=42)
    }
    
    best_rmse = float('inf')
    best_model_name = ""
    best_model_pipe = None
    
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"  Training {name}...")
            if name == "XGBoostRegressor":
                model.fit(X_train_processed, y_train,
                          eval_set=[(X_val_processed, y_val)],
                          verbose=False)
            else:
                model.fit(X_train_processed, y_train)
                
            y_pred = model.predict(X_val_processed)
            
            pipe_to_log = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            
            mlflow.log_params(model.get_params())
            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_r2_score", r2)
            mlflow.log_metric("val_mae", mae)
            
            mlflow.sklearn.log_model(pipe_to_log, f"{name}_model")
            print(f"  {name} model trained. Validation RMSE: {rmse:.2f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_pipe = pipe_to_log
                best_model_name = name

    print(f"--- Best Regression Model: {best_model_name} (RMSE: {best_rmse:.2f}) ---")
    
    print(f"Registering '{best_model_name}' as 'production'...")
    mlflow.sklearn.log_model(
        best_model_pipe,
        "regression_model",
        registered_model_name="EMIPredict_Regression_Model"
    )
    return best_model_pipe

# --- Main execution ---
def main():
    try:
        # --- THIS IS THE FIX ---
        df = load_and_clean_data('emi_prediction_dataset.csv')
        # --- END FIX ---
    except FileNotFoundError:
        print("ERROR: 'emi_prediction_dataset.csv' not found. Please place it in the same folder.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    perform_eda(df)
    
    df_feat = feature_engineer(df)
    
    X = df_feat.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
    y_class = df_feat['emi_eligibility']
    y_reg = df_feat['max_monthly_emi']
    
    categorical_features_list = X.select_dtypes(include='object').columns.tolist()
    for col in categorical_features_list:
        X[col] = X[col].astype(str)
    
    X_train_full, X_test, y_train_class_full, y_test_class, y_train_reg_full, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    X_train, X_val, y_train_class, y_val_class, y_train_reg, y_val_reg = train_test_split(
        X_train_full, y_train_class_full, y_train_reg_full, test_size=0.125, random_state=42, stratify=y_train_class_full 
    )
    
    print(f"Data split complete:")
    print(f"Train set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
    
    preprocessor = get_preprocessor(X)
    preprocessor.fit(X_train_full) 
    
    best_class_model_pipe = train_classification(X_train, X_val, y_train_class, y_val_class, preprocessor)
    best_reg_model_pipe = train_regression(X_train, X_val, y_train_reg, y_val_reg, preprocessor)
    
    print("\n--- Final Evaluation on Unseen Test Set ---")
    
    y_pred_class_test = best_class_model_pipe.predict(X_test)
    
    if isinstance(best_class_model_pipe.named_steps['classifier'], XGBClassifier):
        print("Converting numeric predictions from XGBoost back to string labels for testing...")
        label_map = {'Eligible': 0, 'High_Risk': 1, 'Not_Eligible': 2}
        reverse_map = {v: k for k, v in label_map.items()}
        y_pred_class_test = pd.Series(y_pred_class_test).map(reverse_map)

    f1_test = f1_score(y_test_class, y_pred_class_test, average='weighted')
    print(f"Best Classification Model (Test F1-Score): {f1_test:.4f}")
    
    y_pred_reg_test = best_reg_model_pipe.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_test))
    r2_test = r2_score(y_test_reg, y_pred_reg_test)
    print(f"Best Regression Model (Test RMSE): {rmse_test:.2f}")
    print(f"Best Regression Model (Test R2-Score): {r2_test:.4f}")
    
    print("\nSaving final models trained on all data (Train+Val)...")
    
    y_train_class_to_fit = y_train_class_full
    if isinstance(best_class_model_pipe.named_steps['classifier'], XGBClassifier):
        print("Best classifier is XGBoost, mapping labels for final fit...")
        label_map = {'Eligible': 0, 'High_Risk': 1, 'Not_Eligible': 2}
        y_train_class_to_fit = y_train_class_full.map(label_map)
        
    best_class_model_pipe.fit(X_train_full, y_train_class_to_fit)
    joblib.dump(best_class_model_pipe, 'classification_model.pkl')
    print("Final classification model saved as 'classification_model.pkl'")
    
    best_reg_model_pipe.fit(X_train_full, y_train_reg_full)
    joblib.dump(best_reg_model_pipe, 'regression_model.pkl')
    print("Final regression model saved as 'regression_model.pkl'")
    
    print("\n--- SCRIPT COMPLETE ---")
    print("You can now run 'mlflow ui' to see your experiments and 'streamlit run app.py' to launch the app.")

if __name__ == "__main__":
    main()