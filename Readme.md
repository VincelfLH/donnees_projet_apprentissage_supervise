# README – Dossier "Documents_necessaires_execution"

Ce dossier contient l'ensemble des fichiers essentiels pour exécuter le projet de méta-modélisation sur la base des modèles préalablement entraînés. Il est conçu pour permettre une reproduction complète des prédictions et des évaluations présentées dans le rapport.

## Structure du dossier

### 1. **Jeux de données**
- `jeu_donnees_final.csv` : Jeu d'entraînement final prétraité (fusion complète).
- `jeu_donnees_final_test.csv` : Jeu de test prétraité selon la même structure.
- `jeu_donnees_fusionne.csv` : Fusion intermédiaire des données brutes.

### 2. **Modèles sauvegardés (meta-modèles)**
- `meta_modele_actifs.pkl`, `meta_modele_retraites.pkl` : Modèles stackés optimisés (F1).
- `meta_modele_stacked_optuna.pkl` : Modèle stacké optimisé avec Optuna.

### 3. **Pipelines de prétraitement**
- `pipeline_actifs.pkl`, `pipeline_retraites.pkl` : Pipelines spécifiques par statut.
- `pipeline_modele_*.pkl` : Pipelines des modèles de base (LGBM, ExtraTrees, RegLog, etc.).
- `pretraitement_global.pkl` : Pipeline généralisée pour données globales.

### 4. **Fichiers OOF (Out-of-Fold)**
- `oof_pred_*.npy` : Prédictions OOF pour chaque modèle de base.
- `y_proba_oof_*.npy` : Probabilités prédites utilisées pour l'entraînement du méta-modèle.

### 5. **Fichiers de stacking**
- `X_stack_*.pkl` : Features du méta-modèle (niveaux supérieurs).
- `stack_model_*.pkl` : Modèles stackés enregistrés.

### 6. **Prédictions sur le jeu de test**
- `pred_test_*.npy` : Prédictions binaires ou probabilistes des modèles.
- `predictions_meta_modele_*.xlsx` : Prédictions finales formatées pour exploitation ou soumission.

### 7. **Autres fichiers utiles**
- `statut_*.npy` : Vecteurs indicateurs du statut (actif/retraité).
- `y_*.npy` et `y_*.pkl` : Cibles enregistrées pour chaque étape (OOF, stacking, test).
- `X_train_preprocessed.npy`, `X_test_preprocessed.npy` : Données prétraitées intermédiaires.
- `best_params_lgbm_*.json` : Paramètres optimaux pour LGBM via Optuna.

## Utilisation recommandée
1. Charger les pipelines correspondants selon les statuts.
2. Utiliser les fichiers `X_stack_*.pkl` et `y_*.pkl` pour reconstituer les données d'entraînement.
3. Appliquer les modèles sauvegardés aux données de test (`jeu_donnees_final_test.csv`).
4. Comparer les résultats avec les fichiers de prédiction également fournis.

## Notes
- Tous les fichiers ont été générés dans un environnement Python 3.12.1 avec scikit-learn, lightgbm, imbalanced-learn, pandas, numpy, shap.
- Pour répliquer l'environnement, un fichier `pip install -r requirements.txt` est proposé.

---
Vincent

