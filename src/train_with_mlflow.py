"""
Script d'entraînement avec MLflow pour la prédiction du stress étudiant
Auteur: Hala Hamza
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration MLflow
mlflow.set_experiment("Stress_Level_Prediction")


def load_and_preprocess_data(file_path):
    """
    Charge et prépare les données avec gestion d'erreurs complète
    
    Args:
        file_path: Chemin vers le fichier CSV
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test, scaler, label_encoder)
    """
    
    # 1. Chargement avec gestion d'erreurs
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset chargé: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {file_path}")
        print("\nChemins possibles:")
        print("  - data/StressLevelDataset.csv")
        print("  - ../data/StressLevelDataset.csv")
        print("  - StressLevelDataset.csv")
        raise
    
    # 2. Vérification de la colonne cible
    target_col = 'stress_level'
    if target_col not in df.columns:
        print(f"❌ Colonne '{target_col}' introuvable!")
        print(f"Colonnes disponibles: {df.columns.tolist()}")
        # Chercher une colonne similaire
        similar_cols = [col for col in df.columns if 'stress' in col.lower()]
        if similar_cols:
            print(f"Colonnes similaires trouvées: {similar_cols}")
        raise KeyError(f"Colonne {target_col} introuvable")
    
    # 3. Nettoyage des données
    initial_shape = df.shape
    df = df.dropna()
    df = df.drop_duplicates()
    
    if df.shape != initial_shape:
        print(f"🧹 Nettoyage: {initial_shape} → {df.shape}")
    
    # 4. Analyse de la distribution
    print(f"\n📊 Distribution de {target_col}:")
    print(df[target_col].value_counts().sort_index())
    
    # 5. Vérification du nombre minimum d'échantillons par classe
    min_samples = df[target_col].value_counts().min()
    if min_samples < 2:
        raise ValueError(
            f"❌ Une classe a seulement {min_samples} échantillon(s). "
            "Minimum requis: 2 pour la stratification."
        )
    
    print(f"✅ Toutes les classes ont ≥ {min_samples} échantillons")
    
    # 6. Séparation features/target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"\n📋 Features: {X.shape[1]} colonnes")
    print(f"Colonnes: {X.columns.tolist()[:5]}...")  # Afficher les 5 premières
    
    # 7. Encodage de la variable cible
    le = LabelEncoder()
    
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        print(f"🔤 Encodage des labels textuels: {y.unique()}")
        y_encoded = le.fit_transform(y)
        print(f"   → Encodés en: {np.unique(y_encoded)}")
    else:
        print(f"🔢 Labels déjà numériques: {y.unique()}")
        y_encoded = y.values
        le.classes_ = np.unique(y_encoded)
    
    # 8. Division train/test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    print(f"\n✂️ Split:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test:  {X_test.shape}")
    
    # 9. Normalisation des features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Normalisation appliquée (StandardScaler)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le


def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calcule toutes les métriques de performance
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        y_proba: Probabilités prédites
        
    Returns:
        Dict: Dictionnaire des métriques
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(
            y_true, y_proba, 
            multi_class='ovr', 
            average='weighted'
        )
    }
    return metrics


def train_model_with_mlflow(model, model_name, X_train, X_test, y_train, y_test):
    """
    Entraîne un modèle et log tout dans MLflow
    
    Args:
        model: Instance du modèle scikit-learn
        model_name: Nom du modèle
        X_train, X_test: Features d'entraînement et de test
        y_train, y_test: Labels d'entraînement et de test
        
    Returns:
        Dict: Métriques calculées
    """
    
    with mlflow.start_run(run_name=model_name):
        
        print(f"\n{'='*60}")
        print(f"🤖 Entraînement: {model_name}")
        print(f"{'='*60}")
        
        # 1. Log des paramètres généraux
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        
        # 2. Log des hyperparamètres du modèle
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for param, value in params.items():
                mlflow.log_param(f"model_{param}", value)
        
        # 3. Entraînement
        print("⏳ Entraînement en cours...")
        model.fit(X_train, y_train)
        print("✅ Entraînement terminé")
        
        # 4. Prédictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # 5. Calcul des métriques
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        
        # 6. Affichage des résultats
        print("\n📊 Performances:")
        for metric_name, metric_value in metrics.items():
            print(f"   {metric_name:12s}: {metric_value:.4f}")
            mlflow.log_metric(metric_name, metric_value)
        
        # 7. Rapport de classification détaillé
        class_report = classification_report(y_test, y_pred)
        print(f"\n📋 Rapport de classification:\n{class_report}")
        
        # 8. Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Matrice de confusion:\n{cm}")
        
        # Log de la matrice comme artifact
        mlflow.log_dict(
            {"confusion_matrix": cm.tolist()}, 
            "confusion_matrix.json"
        )
        
        # 9. Sauvegarde du modèle
        # Créer le dossier si nécessaire
        os.makedirs("models", exist_ok=True)
        
        model_path = f"models/{model_name}.pkl"
        joblib.dump(model, model_path)
        
        # 10. Log du modèle dans MLflow
        mlflow.sklearn.log_model(
            model,
            model_name,
            registered_model_name=f"stress_predictor_{model_name}"
        )
        
        # 11. Log du fichier modèle comme artifact
        mlflow.log_artifact(model_path)
        
        print(f"💾 Modèle sauvegardé: {model_path}")
        
        return metrics


def compare_models(results):
    """
    Compare les résultats de tous les modèles
    
    Args:
        results: Dict contenant les métriques de chaque modèle
    """
    print(f"\n{'='*60}")
    print("📊 COMPARAISON DES MODÈLES")
    print(f"{'='*60}\n")
    
    # Créer un DataFrame pour la comparaison
    df_results = pd.DataFrame(results).T
    df_results = df_results.round(4)
    
    print(df_results.to_string())
    
    # Identifier le meilleur modèle selon F1-Score
    best_model_name = df_results['f1_score'].idxmax()
    best_f1 = df_results.loc[best_model_name, 'f1_score']
    
    print(f"\n{'='*60}")
    print(f"🏆 MEILLEUR MODÈLE: {best_model_name}")
    print(f"{'='*60}")
    print(f"F1-Score: {best_f1:.4f}")
    print(f"\nMétriques complètes:")
    for metric, value in df_results.loc[best_model_name].items():
        print(f"   {metric:12s}: {value:.4f}")
    
    return best_model_name, df_results


def main():
    """Pipeline principal d'entraînement avec MLflow"""
    
    print("="*60)
    print("🚀 PIPELINE DE PRÉDICTION DU STRESS ÉTUDIANT")
    print("="*60)
    
    # 1. Chargement et prétraitement des données
    print("\n📂 ÉTAPE 1: Chargement des données")
    print("-" * 60)
    
    # Essayer différents chemins possibles
    possible_paths = [
        'data/StressLevelDataset.csv',
        '../data/StressLevelDataset.csv',
        'StressLevelDataset.csv'
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("❌ Fichier de données introuvable!")
        print("Veuillez placer 'StressLevelDataset.csv' dans:")
        print("  - data/")
        print("  - Le dossier racine du projet")
        return
    
    try:
        X_train, X_test, y_train, y_test, scaler, le = load_and_preprocess_data(data_path)
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement: {e}")
        return
    
    # 2. Sauvegarde des transformateurs
    print("\n💾 Sauvegarde des transformateurs...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("✅ Scaler et Label Encoder sauvegardés")
    
    # 3. Définition des modèles
    print("\n🤖 ÉTAPE 2: Configuration des modèles")
    print("-" * 60)
    
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision_Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    print(f"Modèles à entraîner: {list(models.keys())}")
    
    # 4. Entraînement des modèles
    print("\n🔄 ÉTAPE 3: Entraînement avec MLflow")
    print("-" * 60)
    
    results = {}
    
    for model_name, model in models.items():
        try:
            metrics = train_model_with_mlflow(
                model, model_name, X_train, X_test, y_train, y_test
            )
            results[model_name] = metrics
        except Exception as e:
            print(f"\n❌ Erreur avec {model_name}: {e}")
            continue
    
    # 5. Comparaison finale
    if results:
        print("\n📈 ÉTAPE 4: Analyse comparative")
        print("-" * 60)
        best_model_name, df_results = compare_models(results)
        
        # 6. Log du meilleur modèle dans un run séparé
        with mlflow.start_run(run_name="Best_Model_Summary"):
            mlflow.log_param("best_model", best_model_name)
            for metric, value in results[best_model_name].items():
                mlflow.log_metric(f"best_{metric}", value)
        
        # 7. Sauvegarde du rapport
        report_path = 'models/training_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("RAPPORT D'ENTRAÎNEMENT - PRÉDICTION DU STRESS\n")
            f.write("="*60 + "\n\n")
            f.write(df_results.to_string())
            f.write(f"\n\n{'='*60}\n")
            f.write(f"MEILLEUR MODÈLE: {best_model_name}\n")
            f.write(f"{'='*60}\n")
            for metric, value in results[best_model_name].items():
                f.write(f"{metric:12s}: {value:.4f}\n")
        
        print(f"\n💾 Rapport sauvegardé: {report_path}")
    
    # 8. Instructions finales
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("="*60)
    print("\n📊 Pour visualiser les résultats dans MLflow UI:")
    print("   1. Ouvrez un nouveau terminal")
    print("   2. Exécutez: mlflow ui")
    print("   3. Ouvrez: http://localhost:5000")
    print("\n💡 Les modèles sont sauvegardés dans: models/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()