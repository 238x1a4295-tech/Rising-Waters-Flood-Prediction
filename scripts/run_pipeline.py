import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import os

# Create docs/plots directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(BASE_DIR, '../docs/plots')
os.makedirs(plots_dir, exist_ok=True)

# ==========================================
# PHASE 1: Environment Setup & Data Exploration
# ==========================================

def generate_and_read_dataset(filepath=None):
    if filepath is None:
        filepath = os.path.join(BASE_DIR, '../data/raw/flood dataset.xlsx')
    print(f"Reading provided dataset from {filepath}...")
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading dataset. Please make sure openpyxl is installed: {e}")
        raise e
        
    print("Applying proper cleaning and formatting...")
    # Rename columns to match the features our model and app expect
    df = df.rename(columns={
        'Temp': 'Temperature',
        'Humidity': 'Humidity',
        'Cloud Cover': 'Cloud_Visibility',
        'ANNUAL': 'Annual_Rainfall',
        'Jun-Sep': 'Seasonal_Rainfall',
        'flood': 'class'
    })
    
    # Drop unused columns
    cols_to_drop = ['Jan-Feb', 'Mar-May', 'Oct-Dec', 'avgjune', 'sub']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Save to CSV for reference
    processed_path = os.path.join(BASE_DIR, '../data/processed/flood_dataset.csv')
    df.to_csv(processed_path, index=False)
    print(f"Dataset processed and saved to {processed_path}")
    
    return df

def perform_eda(df):
    print("\n--- Descriptive Analysis ---")
    print(df.info())
    print("\n", df.describe())
    
    print("\n--- Multivariate Analysis ---")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.pairplot(df, hue='class', diag_kind='kde')
    plt.savefig(os.path.join(plots_dir, 'pairplot.png'))
    plt.close()
    print(f"Saved EDA plots to {plots_dir}")

def handle_missing_values(df):
    print("\n--- Handling Missing Values ---")
    print("Missing values before:", df.isnull().sum().to_dict())
    
    # Impute missing values with median
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    print("Missing values after:", df.isnull().sum().to_dict())
    return df

# ==========================================
# PHASE 2: Data Preprocessing & Splitting
# ==========================================

def handle_outliers(df):
    print("\n--- Handling Outliers (IQR Capping) ---")
    features = ['Annual_Rainfall', 'Cloud_Visibility', 'Seasonal_Rainfall', 'Temperature', 'Humidity']
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Capping
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

def preprocess_and_split(df):
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    transform_path = os.path.join(BASE_DIR, '../models/transform.save')
    joblib.dump(sc, transform_path)
    print(f"Saved scaler to {transform_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ==========================================
# PHASE 3 & 4: Model Training, Evaluation & Comparison
# ==========================================

model_results = {}

def decisiontree(X_train, X_test, y_train, y_test):
    print("\n--- Decision Tree Model ---")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    model_results['Decision Tree'] = acc
    print(f"Accuracy: {acc:.4f}")
    return dt

def randomForest(X_train, X_test, y_train, y_test):
    print("\n--- Random Forest Model ---")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    model_results['Random Forest'] = acc
    print(f"Accuracy: {acc:.4f}")
    return rf

def KNN(X_train, X_test, y_train, y_test):
    print("\n--- KNN Model ---")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    model_results['KNN'] = acc
    print(f"Accuracy: {acc:.4f}")
    return knn

def xgboost_model(X_train, X_test, y_train, y_test):
    print("\n--- XGBoost Model ---")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    model_results['XGBoost'] = acc
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return xgb

def compareModel():
    print("\n--- Model Comparison ---")
    print(f"{'Model':<15} | {'Accuracy':<10}")
    print("-" * 28)
    for model, acc in sorted(model_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:<15} | {acc:.4f}")

if __name__ == "__main__":
    df = generate_and_read_dataset()
    perform_eda(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    
    _ = decisiontree(X_train, X_test, y_train, y_test)
    rf = randomForest(X_train, X_test, y_train, y_test)
    _ = KNN(X_train, X_test, y_train, y_test)
    xgb = xgboost_model(X_train, X_test, y_train, y_test)
    
    compareModel()
    
    # Milestone 15: Saving the best model
    model_save_path = os.path.join(BASE_DIR, '../models/floods.save')
    joblib.dump(rf, model_save_path)
    print(f"\nSaved Random Forest model (balanced) to {model_save_path}")
    print("\nPipeline execution complete!")
