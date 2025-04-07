import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay
import shap
import json

def load_data(directory):
    all_data = []
    for file in os.listdir(directory):
        if file.endswith('_fermentation_data.xlsx'):
            try:
                df = pd.read_excel(os.path.join(directory, file))
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
    if not all_data:
        raise ValueError("No valid datasets found in the directory.")
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(df):
    df = df.dropna(subset=['Phase'])
    df = df.ffill().bfill()
    return df

def perform_hyperparameter_tuning(X_train, y_train):
    param_dist = {
        'n_estimators': [50, 100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, cv=3, n_iter=20, n_jobs=-1, verbose=2)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

def plot_feature_importance(model, feature_names, save_path):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[sorted_indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in sorted_indices], rotation=45, ha='right')
    plt.xlabel("Feature Names")
    plt.ylabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"1_Feature_importances_{datetime.now().strftime('%d%m%Y')}.png"))
    plt.close()

def save_model(model, filename):
    joblib.dump(model, filename)

def evaluate_model(model, X_test, y_test, base_path):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()  # Ensure a new figure is created
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.savefig(os.path.join(base_path, f"2_Confusion_matrix_{datetime.now().strftime('%d%m%Y')}.png"))
    plt.close()  # Prevent overlapping figures
    return accuracy, precision, recall, f1

def balance_classes(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

def select_features(model, X_train, X_test, y_train):
    selector = SelectFromModel(model, threshold="median", prefit=False)
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()]
    return X_train_selected, X_test_selected, selected_features

def explain_model(model, X_train, save_path):
    background = shap.sample(X_train, 100)
    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X_train[:500])
    # Assuming you want the SHAP values for the first class
    shap_values_for_plot = shap_values[:, :, 0]
    plt.figure()  # Ensure a new figure is created
    shap.summary_plot(shap_values_for_plot, X_train.iloc[:500, :], show=False)
    plt.savefig(os.path.join(save_path, f"3_SHAP_summary_{datetime.now().strftime('%d%m%Y')}.png"))
    plt.close()  # Prevent overlapping figures

def load_model(filename):
    return joblib.load(filename)

def main():
    base_path = r'C:\Users\GeorgiosBalamotis\sqale.ai\3-PRODUCT DEVELOPMENT - PROJECT-101-eNose - PROJECT-101-eNose\2024 12 16-Experimental data generated\Fermentations_DL'
    os.makedirs(base_path, exist_ok=True)
    data = load_data(r'C:\Users\GeorgiosBalamotis\sqale.ai\3-PRODUCT DEVELOPMENT - PROJECT-101-eNose - PROJECT-101-eNose\2024 12 16-Experimental data generated\Fermentations_DL\Datasets')
    data = preprocess_data(data)
    features = data[['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4']]
    labels = data['Phase']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model_filename = os.path.join(base_path, 'Trained_RF_model.joblib')
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    else:
        model = perform_hyperparameter_tuning(X_train, y_train)
    X_train, y_train = balance_classes(X_train, y_train)
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test, base_path)
    explain_model(model, X_train, base_path)
    save_model(model, model_filename)
    report_filename = os.path.join(base_path, f"4_Classification_report_{datetime.now().strftime('%d%m%Y')}.json")
    report_data = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    with open(report_filename, 'w') as f:
        json.dump(report_data, f, indent=4)
    plot_feature_importance(model, features.columns, base_path)

if __name__ == "__main__":
    main()
