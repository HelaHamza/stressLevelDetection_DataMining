import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report, 
    confusion_matrix,
    roc_auc_score
)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Entraîne 3 modèles et évalue leurs performances selon plusieurs métriques.
    Génère des visualisations et identifie le meilleur modèle.
    """
    
    # Création du dossier results avec chemin absolu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print(f"📁 Dossier de sauvegarde : {results_dir}\n")

    # Définition des modèles
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Stockage des résultats
    results_dict = {}
    metrics_df_data = []

    print("="*70)
    print("ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
    print("="*70)

    for name, model in models.items():
        print(f"\n🔹 Entraînement : {name}")
        
        # Entraînement
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC (si classification binaire ou multi-classe)
        try:
            if len(np.unique(y_test)) == 2:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                y_pred_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = None
        
        # Stockage des métriques
        results_dict[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        metrics_df_data.append({
            'Modèle': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc if roc_auc else 'N/A'
        })
        
        # Affichage des résultats
        print(f"   ✓ Accuracy:  {accuracy:.4f}")
        print(f"   ✓ Precision: {precision:.4f}")
        print(f"   ✓ Recall:    {recall:.4f}")
        print(f"   ✓ F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"   ✓ ROC-AUC:   {roc_auc:.4f}")
        
        # Rapport de classification détaillé
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': 'Count'})
        plt.title(f"Matrice de Confusion - {name}", fontsize=14, fontweight='bold')
        plt.xlabel("Classe Prédite", fontsize=12)
        plt.ylabel("Classe Réelle", fontsize=12)
        plt.tight_layout()
        cm_path = os.path.join(results_dir, f"confusion_matrix_{name.replace(' ', '_')}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"   ✓ Matrice sauvegardée : {cm_path}")

    # Création du DataFrame des métriques
    metrics_df = pd.DataFrame(metrics_df_data)
    
    # Sauvegarde des métriques dans un fichier CSV
    csv_path = os.path.join(results_dir, "metrics_comparison.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    # Identification du meilleur modèle
    print("\n" + "="*70)
    print("COMPARAISON DES MODÈLES")
    print("="*70)
    print("\n📊 Tableau récapitulatif :")
    print(metrics_df.to_string(index=False))
    
    # Déterminer le meilleur modèle (basé sur F1-Score)
    best_model_name = max(results_dict.items(), key=lambda x: x[1]['f1_score'])[0]
    best_metrics = results_dict[best_model_name]
    
    print(f"\n🏆 MEILLEUR MODÈLE : {best_model_name}")
    print(f"   → F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"   → Accuracy: {best_metrics['accuracy']:.4f}")
    
    # Visualisation 1 : Comparaison par métrique
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparaison des Modèles - Toutes Métriques', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for idx, (metric, metric_name, color) in enumerate(zip(metrics_to_plot, metric_names, colors)):
        ax = axes[idx // 2, idx % 2]
        values = [results_dict[model][metric] for model in models.keys()]
        bars = ax.bar(models.keys(), values, color=color, alpha=0.7, edgecolor='black')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} par Modèle', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(models.keys(), rotation=15, ha='right')
    
    plt.tight_layout()
    comparison_path = os.path.join(results_dir, "metrics_comparison.png")
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"\n✓ Graphique comparatif sauvegardé : {comparison_path}")
    
    # Visualisation 2 : Radar chart pour comparaison globale
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    
    for name, color in zip(models.keys(), ['#3498db', '#2ecc71', '#e74c3c']):
        values = [
            results_dict[name]['accuracy'],
            results_dict[name]['precision'],
            results_dict[name]['recall'],
            results_dict[name]['f1_score']
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Comparaison Globale des Performances', y=1.08, fontsize=14, fontweight='bold')
    ax.grid(True)
    
    plt.tight_layout()
    radar_path = os.path.join(results_dir, "radar_comparison.png")
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique radar sauvegardé : {radar_path}")
    
    # Sauvegarde du rapport final
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT D'ÉVALUATION DES MODÈLES\n")
        f.write("="*70 + "\n\n")
        
        for name in models.keys():
            f.write(f"\n{name}:\n")
            f.write(f"  - Accuracy:  {results_dict[name]['accuracy']:.4f}\n")
            f.write(f"  - Precision: {results_dict[name]['precision']:.4f}\n")
            f.write(f"  - Recall:    {results_dict[name]['recall']:.4f}\n")
            f.write(f"  - F1-Score:  {results_dict[name]['f1_score']:.4f}\n")
            if results_dict[name]['roc_auc']:
                f.write(f"  - ROC-AUC:   {results_dict[name]['roc_auc']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"MEILLEUR MODÈLE: {best_model_name}\n")
        f.write(f"F1-Score: {best_metrics['f1_score']:.4f}\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Rapport textuel sauvegardé : {report_path}")
    print(f"\n✅ Tous les résultats ont été sauvegardés dans : {results_dir}/")
    
    return results_dict, best_model_name