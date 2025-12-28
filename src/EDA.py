import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda(data_path):
    """
    Effectue une analyse exploratoire complète des données
    et sauvegarde les visualisations dans results/eda/
    """
    df = pd.read_csv(data_path)
    
    # Création du dossier pour les résultats EDA
    # Utiliser le chemin absolu basé sur le script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    eda_dir = os.path.join(project_root, "results", "eda")
    
    if not os.path.exists(eda_dir):
        os.makedirs(eda_dir)
    print(f"📁 Dossier de sauvegarde : {eda_dir}")

    print("\n🔹 Aperçu des données")
    print(df.head())

    print("\n🔹 Informations générales")
    print(df.info())

    print("\n🔹 Statistiques descriptives")
    print(df.describe())
    
    # Vérification des valeurs manquantes
    print("\n🔹 Valeurs manquantes")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ Aucune valeur manquante détectée")
    else:
        print(missing[missing > 0])
    
    # Distribution du stress_level
    print("\n🔹 Distribution de la variable cible (stress_level)")
    print(df['stress_level'].value_counts().sort_index())
    
    # Visualisation 1 : Distribution du stress_level
    plt.figure(figsize=(8, 5))
    sns.countplot(x="stress_level", data=df, palette="viridis")
    plt.title("Distribution du niveau de stress", fontsize=14, fontweight='bold')
    plt.xlabel("Niveau de stress", fontsize=12)
    plt.ylabel("Nombre d'observations", fontsize=12)
    for container in plt.gca().containers:
        plt.gca().bar_label(container)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "stress_distribution.png"), dpi=300)
    plt.close()
    print(f"   ✓ Graphique sauvegardé : {os.path.join(eda_dir, 'stress_distribution.png')}")
    
    # Visualisation 2 : Matrice de corrélation
    plt.figure(figsize=(14, 10))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0, square=True, linewidths=0.5)
    plt.title("Matrice de corrélation des variables", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "correlation_matrix.png"), dpi=300)
    plt.close()
    print(f"   ✓ Graphique sauvegardé : {os.path.join(eda_dir, 'correlation_matrix.png')}")
    
    # Visualisation 3 : Top corrélations avec stress_level
    stress_corr = correlation['stress_level'].sort_values(ascending=False)[1:11]
    plt.figure(figsize=(10, 6))
    stress_corr.plot(kind='barh', color='steelblue')
    plt.title("Top 10 - Corrélations avec le niveau de stress", fontsize=14, fontweight='bold')
    plt.xlabel("Coefficient de corrélation", fontsize=12)
    plt.ylabel("Variables", fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "stress_correlations.png"), dpi=300)
    plt.close()
    print(f"   ✓ Graphique sauvegardé : {os.path.join(eda_dir, 'stress_correlations.png')}")
    
    # Visualisation 4 : Distributions des principales variables
    main_features = ['anxiety_level', 'self_esteem', 'depression', 'sleep_quality', 
                     'academic_performance', 'social_support']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution des principales variables', fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(main_features):
        ax = axes[idx // 3, idx % 3]
        sns.histplot(df[feature], kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "features_distribution.png"), dpi=300)
    plt.close()
    print(f"   ✓ Graphique sauvegardé : {os.path.join(eda_dir, 'features_distribution.png')}")
    
    # Visualisation 5 : Boxplots par niveau de stress
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparaison des variables par niveau de stress', fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(main_features):
        ax = axes[idx // 3, idx % 3]
        sns.boxplot(x='stress_level', y=feature, data=df, ax=ax, palette='Set2')
        ax.set_title(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('Niveau de stress')
        ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "features_by_stress.png"), dpi=300)
    plt.close()
    print(f"   ✓ Graphique sauvegardé : {os.path.join(eda_dir, 'features_by_stress.png')}")
    
    # Sauvegarde des statistiques descriptives
    stats_path = os.path.join(eda_dir, "statistics_summary.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ANALYSE EXPLORATOIRE DES DONNÉES - RÉSUMÉ\n")
        f.write("="*70 + "\n\n")
        
        f.write("Dimensions du dataset:\n")
        f.write(f"  - Nombre de lignes: {df.shape[0]}\n")
        f.write(f"  - Nombre de colonnes: {df.shape[1]}\n\n")
        
        f.write("Distribution de stress_level:\n")
        f.write(df['stress_level'].value_counts().sort_index().to_string())
        f.write("\n\n")
        
        f.write("Top 10 corrélations avec stress_level:\n")
        f.write(stress_corr.to_string())
        f.write("\n\n")
        
        f.write("Statistiques descriptives:\n")
        f.write(df.describe().to_string())
    
    print(f"   ✓ Statistiques sauvegardées : {stats_path}")
    
    print(f"\n✅ Tous les fichiers EDA ont été sauvegardés dans : {eda_dir}/")
    
    return df