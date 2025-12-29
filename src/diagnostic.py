import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv('data/StressLevelDataset.csv')

print("=" * 60)
print("DIAGNOSTIC DU DATASET")
print("=" * 60)

# 1. Informations générales
print("\n📊 INFORMATIONS GÉNÉRALES")
print(f"Shape: {df.shape}")
print(f"Colonnes: {df.columns.tolist()}")

# 2. Analyse de la colonne stress_level
print("\n🎯 ANALYSE DE 'stress_level'")
print(f"Type de données: {df['stress_level'].dtype}")
print(f"Valeurs uniques: {df['stress_level'].unique()}")
print(f"Nombre de valeurs uniques: {df['stress_level'].nunique()}")

# 3. Distribution
print("\n📈 DISTRIBUTION DES CLASSES")
print(df['stress_level'].value_counts().sort_index())

# 4. Vérifier les valeurs manquantes
print("\n❓ VALEURS MANQUANTES")
print(f"Dans stress_level: {df['stress_level'].isna().sum()}")
print(f"Total dataset: {df.isna().sum().sum()}")

# 5. Vérifier les doublons
print("\n🔄 DOUBLONS")
print(f"Lignes dupliquées: {df.duplicated().sum()}")

# 6. Premières lignes
print("\n👀 PREMIÈRES LIGNES")
print(df.head())

# 7. Vérifier si la colonne existe bien
print("\n🔍 COLONNES DU DATASET")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col} ({df[col].dtype})")

# 8. Solution recommandée
print("\n" + "=" * 60)
print("💡 SOLUTION")
print("=" * 60)

if df['stress_level'].dtype == 'object' or isinstance(df['stress_level'].iloc[0], str):
    print("✓ Votre colonne contient des valeurs TEXTUELLES")
    print("  → Il faut utiliser LabelEncoder")
else:
    print("✓ Votre colonne contient des valeurs NUMÉRIQUES")
    print("  → Pas besoin de LabelEncoder")

# Vérifier si les classes sont équilibrées
min_class_size = df['stress_level'].value_counts().min()
if min_class_size < 2:
    print(f"\n⚠️ PROBLÈME DÉTECTÉ: Une classe a seulement {min_class_size} échantillon(s)")
    print("   Cela empêche la stratification lors du train_test_split")
else:
    print(f"\n✓ Toutes les classes ont au moins {min_class_size} échantillons")