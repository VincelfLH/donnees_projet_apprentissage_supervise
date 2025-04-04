import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# ===============================
# 1) Chargement du jeu de données
# ===============================
def load_data_segmented(file_name="jeu_donnees_final.csv"):
    chemins_possibles = [
        r"C:/Users/vince/Documents/Université PSL/Paris_Dauphine-PSL/Apprentissage_supervisé/Projet/donnees_projet_apprentissage_supervise",
        os.path.join(os.getcwd(), "project-12-files")
    ]
    for chemin in chemins_possibles:
        chemin_fichier = os.path.join(chemin, file_name)
        if os.path.exists(chemin_fichier):
            print(f"✅ Fichier chargé localement : {chemin_fichier}")
            return pd.read_csv(chemin_fichier)

    # Fallback GitHub
    url = "https://raw.githubusercontent.com/VinceflLH/donnees_projet_apprentissage_supervise/main/jeu_donnees_final.csv"
    try:
        print(f"🌐 Chargement depuis GitHub : {url}")
        return pd.read_csv(url)
    except Exception as e:
        raise FileNotFoundError("❌ Échec du chargement : ni localement, ni via GitHub.") from e

# ===============================
# 2) Construction des pipelines segmentées
# ===============================
def construire_pipelines_segmentees(df):
    colonnes_exclure = [
        "Unique_id", "target", "Job_categor", "Region", 
        "job_42_regroupe", "job_desc_n2", "retirement_age", "former_job_42"
    ]

    df_actifs = df[df["Statut"] == "Actif"].copy()
    df_retraites = df[df["Statut"] == "Retraité"].copy()

    X_actifs = df_actifs.drop(columns=colonnes_exclure)
    y_actifs = df_actifs["target"]

    X_retraites = df_retraites.drop(columns=colonnes_exclure)
    y_retraites = df_retraites["target"]

    for col in X_actifs.select_dtypes(include=["object", "category"]).columns:
        X_actifs[col] = X_actifs[col].astype(str)
    for col in X_retraites.select_dtypes(include=["object", "category"]).columns:
        X_retraites[col] = X_retraites[col].astype(str)

    def filtrer_colonnes_existantes(liste, df):
        return [col for col in liste if col in df.columns]

    # === ACTIFS
    numeriques_actifs_knn = filtrer_colonnes_existantes(["Remuneration", "working_hours"], X_actifs)
    numeriques_actifs_0 = filtrer_colonnes_existantes(["distance_job_km"], X_actifs)

    cat_actifs = [
        "COMPANY_CATEGORY", "EMPLOYEE_COUNT", "JOB_CONDITION", 
        "activity_sector", "Emp_contract"
    ]
    cat_actifs += [col for col in X_actifs.select_dtypes(include=["object", "category"]).columns
                   if col not in cat_actifs and col not in numeriques_actifs_knn]
    cat_actifs = filtrer_colonnes_existantes(cat_actifs, X_actifs)

    pipeline_actifs = ColumnTransformer([
        ("knn_impute", Pipeline([
            ("impute", KNNImputer(n_neighbors=5)),
            ("scale", StandardScaler())
        ]), numeriques_actifs_knn),

        ("zero_impute", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", StandardScaler())
        ]), numeriques_actifs_0),

        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Non concerné")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_actifs)
    ])

    # === RETRAITÉS
    numeriques_retraites_knn = filtrer_colonnes_existantes(["retirement_pay"], X_retraites)
    numeriques_retraites_0 = filtrer_colonnes_existantes(["distance_former_km"], X_retraites)

    cat_retraites = ["Former_emp_contract", "Former_job_42"]
    cat_retraites += [col for col in X_retraites.select_dtypes(include=["object", "category"]).columns
                      if col not in cat_retraites and col not in numeriques_retraites_knn]
    cat_retraites = filtrer_colonnes_existantes(cat_retraites, X_retraites)

    pipeline_retraites = ColumnTransformer([
        ("knn_impute", Pipeline([
            ("impute", KNNImputer(n_neighbors=5)),
            ("scale", StandardScaler())
        ]), numeriques_retraites_knn),

        ("zero_impute", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", StandardScaler())
        ]), numeriques_retraites_0),

        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Non concerné")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_retraites)
    ])

    return X_actifs, y_actifs, pipeline_actifs, X_retraites, y_retraites, pipeline_retraites

# ===============================
# 3) Fonction principale (optionnelle)
# ===============================
if __name__ == '__main__':
    df = load_data_segmented()
    X_actifs, y_actifs, pipeline_actifs, X_retraites, y_retraites, pipeline_retraites = construire_pipelines_segmentees(df)

    pipeline_actifs.fit(X_actifs)
    pipeline_retraites.fit(X_retraites)

    chemin_sauvegarde = r"C:/Users/vince/Documents/Université PSL/Paris_Dauphine-PSL/Apprentissage_supervisé/Projet/donnees_projet_apprentissage_supervise"
    joblib.dump(pipeline_actifs, os.path.join(chemin_sauvegarde, "pipeline_actifs.pkl"))
    joblib.dump(pipeline_retraites, os.path.join(chemin_sauvegarde, "pipeline_retraites.pkl"))

    print("✅ Pipelines segmentées sauvegardées avec succès.")