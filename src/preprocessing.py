import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

def preprocess_data(data_path):
    """
    Effectue le prétraitement complet des données :
    - Chargement
    - Nettoyage
    - Encodage
    - Normalisation
    - Split train/test
    
    Sauvegarde également le scaler et l'encoder pour utilisation future
    """
    print("📂 Chargement des données...")
    df = pd.read_csv(data_path)
    print(f"   ✓ {df.shape[0]} lignes et {df.shape[1]} colonnes chargées")
    
    # Vérification des valeurs manquantes
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"⚠️  {missing_before} valeurs manquantes détectées")
        df = df.dropna()
        print(f"   ✓ Lignes supprimées, nouveau total: {df.shape[0]}")
    else:
        print("   ✓ Aucune valeur manquante")
    
    # Vérification des doublons
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  {duplicates} doublons détectés")
        df = df.drop_duplicates()
        print(f"   ✓ Doublons supprimés, nouveau total: {df.shape[0]}")
    else:
        print("   ✓ Aucun doublon")

    # Encodage de la variable cible
    print("\n🔄 Encodage de la variable cible (stress_level)...")
    encoder = LabelEncoder()
    df['stress_level'] = encoder.fit_transform(df['stress_level'])
    print(f"   ✓ Classes: {encoder.classes_}")
    print(f"   ✓ Distribution après encodage:")
    for class_label, encoded_value in zip(encoder.classes_, range(len(encoder.classes_))):
        count = (df['stress_level'] == encoded_value).sum()
        print(f"      - Classe {class_label} → {encoded_value} ({count} échantillons)")

    # Séparation X / y
    print("\n✂️  Séparation des features et de la cible...")
    X = df.drop('stress_level', axis=1)
    y = df['stress_level']
    print(f"   ✓ X shape: {X.shape}")
    print(f"   ✓ y shape: {y.shape}")

    # Encodage des variables catégorielles (si présentes)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"\n🔤 Encodage des variables catégorielles: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f"   ✓ Nouvelles features après encodage: {X.shape[1]}")
    else:
        print("\n   ℹ️  Aucune variable catégorielle à encoder")

    # Normalisation
    print("\n📊 Normalisation des données (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   ✓ Moyenne après scaling: {X_scaled.mean():.6f}")
    print(f"   ✓ Écart-type après scaling: {X_scaled.std():.6f}")

    # Train / Test split
    print("\n🎲 Division train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✓ Train set: {X_train.shape[0]} échantillons ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
    print(f"   ✓ Test set:  {X_test.shape[0]} échantillons ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")
    
    # Distribution des classes dans train/test
    print("\n📈 Distribution des classes:")
    print("   Train:")
    for class_val in np.unique(y_train):
        count = (y_train == class_val).sum()
        print(f"      - Classe {class_val}: {count} ({count/len(y_train)*100:.1f}%)")
    print("   Test:")
    for class_val in np.unique(y_test):
        count = (y_test == class_val).sum()
        print(f"      - Classe {class_val}: {count} ({count/len(y_test)*100:.1f}%)")
    
    # Sauvegarde du scaler et de l'encoder pour utilisation future
    # CORRECTION : Utiliser le chemin absolu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_dir = os.path.join(project_root, "results", "models")
    
    # IMPORTANT : Créer le dossier s'il n'existe pas
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"\n📁 Dossier créé : {models_dir}")
    
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    print(f"\n💾 Scaler sauvegardé: {scaler_path}")
    print(f"💾 Encoder sauvegardé: {encoder_path}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()