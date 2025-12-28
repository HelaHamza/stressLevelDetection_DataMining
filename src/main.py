import os
import warnings
warnings.filterwarnings('ignore')

from EDA import run_eda
from preprocessing import preprocess_data
from modeling import train_and_evaluate

# Chemin vers le dataset
DATA_PATH = "data/StressLevelDataset.csv"
#DATA_PATH="data\Student_Mental_health.csv"

def main():
    print("\n" + "🎯"*35)
    print("   PROJET DATA MINING - CLASSIFICATION DU NIVEAU DE STRESS")
    print("🎯"*35 + "\n")
    
    # ÉTAPE 1 : Analyse exploratoire
    print("\n" + "="*70)
    print("ÉTAPE 1 : ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    print("="*70)
    try:
        run_eda(DATA_PATH)
        print("✅ EDA terminée avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de l'EDA : {e}")
        return
    
    # ÉTAPE 2 : Prétraitement
    print("\n" + "="*70)
    print("ÉTAPE 2 : PRÉTRAITEMENT DES DONNÉES")
    print("="*70)
    try:
        result = preprocess_data(DATA_PATH)
        if len(result) == 5:
            X_train, X_test, y_train, y_test, feature_names = result
        else:
            X_train, X_test, y_train, y_test = result
            feature_names = None
        print(f"\n✅ Prétraitement terminé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du prétraitement : {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ÉTAPE 3 : Modélisation et évaluation
    print("\n" + "="*70)
    print("ÉTAPE 3 : MODÉLISATION ET ÉVALUATION")
    print("="*70)
    try:
        results_dict, best_model_name = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Résumé final
        print("\n" + "="*70)
        print("RÉSUMÉ FINAL")
        print("="*70)
        print("\n📈 Performances de tous les modèles :\n")
        
        for model_name, metrics in results_dict.items():
            symbol = "🏆" if model_name == best_model_name else "  "
            print(f"{symbol} {model_name:20s} | F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")
        
        print(f"\n🎉 Le modèle recommandé est : {best_model_name}")
        print(f"   Avec un F1-Score de {results_dict[best_model_name]['f1_score']:.4f}")
        
        print("\n📁 Fichiers générés dans results/ :")
        print("   - confusion_matrix_*.png : Matrices de confusion")
        print("   - metrics_comparison.png : Comparaison des métriques")
        print("   - radar_comparison.png : Graphique radar")
        print("   - metrics_comparison.csv : Données des métriques")
        print("   - evaluation_report.txt : Rapport détaillé")
        
    except Exception as e:
        print(f"❌ Erreur lors de la modélisation : {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "✅"*35)
    print("   PROJET TERMINÉ AVEC SUCCÈS")
    print("✅"*35 + "\n")

if __name__ == "__main__":
    main()