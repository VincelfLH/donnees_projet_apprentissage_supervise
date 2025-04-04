import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ========================
# 1. Chargement des données
# ========================
def load_data(file_name="jeu_donnees_final.csv", low_memory=False):
    chemins_locaux = [
        r"C:/Users/vince/Documents/Université PSL/Paris_Dauphine-PSL/Apprentissage_supervisé/Projet/donnees_projet_apprentissage_supervise",
        os.path.join(os.getcwd(), "project-12-files")
    ]
    jeu_donnees = None
    chemin_final = None

    for chemin in chemins_locaux:
        chemin_fichier = os.path.join(chemin, file_name)
        if os.path.exists(chemin_fichier):
            print(f"✅ Chargement local depuis : {chemin_fichier}")
            jeu_donnees = pd.read_csv(chemin_fichier, low_memory=low_memory)
            chemin_final = chemin
            break

    if jeu_donnees is None and file_name == "jeu_donnees_final.csv":
        url_github = "https://raw.githubusercontent.com/VinceflLH/donnees_projet_apprentissage_supervise/main/jeu_donnees_final.csv"
        try:
            print(f"🌐 Chargement depuis GitHub : {url_github}")
            jeu_donnees = pd.read_csv(url_github, low_memory=low_memory)
            chemin_final = url_github
        except Exception as e:
            raise FileNotFoundError("❌ Impossible de charger le fichier, ni localement ni via GitHub.") from e

    if jeu_donnees is None:
        raise FileNotFoundError(f"❌ Fichier non trouvé : {file_name}")

    return jeu_donnees, chemin_final

# ========================
# 2. Préparation des données
# ========================
def prepare_data(df):
    # Ajout des flags de missing
    df["Remuneration_missing"] = df["Remuneration"].isna().astype(int)
    df["distance_job_km_missing"] = df["distance_job_km"].isna().astype(int)
    df["retirement_pay_missing"] = df["retirement_pay"].isna().astype(int)
    df["distance_former_km_missing"] = df["distance_former_km"].isna().astype(int)
    df["Statut_encoded"] = (df["Statut"] == "Actif").astype(int)
    
    # Ajout des flags de missing pour les variables catégorielles segmentées pouvant contenir des valeurs manquantes
    colonnes_cat_avec_valeurs_manquantes = [
        "Emp_contract", "COMPANY_CATEGORY", "EMPLOYEE_COUNT", "Contract_type",
        "activity_sector", "JOB_CONDITION", "Job_dep", "Former_dep",
        "Former_emp_contract",
    ]
    for col in colonnes_cat_avec_valeurs_manquantes:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Correction des colonnes numériques
    for col in ["Remuneration", "retirement_pay"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
           
    # Split X / y
    X = df.drop(columns=["Unique_id", "target", "job_desc_n1", "job_desc_n2", "Activity_type", "Region", "Job_categor", "retirement_age", "former_job_42"]).copy()
    y = df["target"].copy()

    # Encodage binaire explicite de la cible
    y = y.map({"L": 0, "T": 1})
    
    return X, y

def harmoniser_data_test(df):
    df_original = df.copy()  # ← pour conserver "Unique_id" si elle existe

    df["Remuneration_missing"] = df["Remuneration"].isna().astype(int)
    df["distance_job_km_missing"] = df["distance_job_km"].isna().astype(int)
    df["retirement_pay_missing"] = df["retirement_pay"].isna().astype(int)
    df["distance_former_km_missing"] = df["distance_former_km"].isna().astype(int)
    df["Statut_encoded"] = (df["Statut"] == "Actif").astype(int)
    
    colonnes_cat_avec_valeurs_manquantes = [
        "Emp_contract", "COMPANY_CATEGORY", "EMPLOYEE_COUNT", "Contract_type",
        "activity_sector", "JOB_CONDITION", "Job_dep", "Former_dep",
        "Former_emp_contract",
    ]
    for col in colonnes_cat_avec_valeurs_manquantes:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    for col in ["Remuneration", "retirement_pay"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    colonnes_a_supprimer = ["Unique_id", "target", "job_desc_n1", "job_desc_n2", "Activity_type", "Region", "Job_categor", "retirement_age", "former_job_42"]
    df = df.drop(columns=[col for col in colonnes_a_supprimer if col in df.columns])

    # Ajout de la colonne Unique_id à la fin
    if "Unique_id" in df_original.columns:
        df["Unique_id"] = df_original["Unique_id"].values

    return df

# ========================
# 3. Préprocesseur
# ========================
def to_str(df):
    return df.astype(str)

def pipeline_knn():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler())
    ])

def build_global_preprocessor(X):
    # Détection des colonnes numériques et catégorielles
    colonnes_num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    colonnes_cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Variables spécifiques selon le statut
    colonnes_actifs_uniques = [
        "Remuneration", "Emp_contract", "working_hours", "Job_dep", "COMPANY_CATEGORY", "Contract_type",
        "EMPLOYEE_COUNT", "JOB_CONDITION", "activity_sector", "distance_job_km"
    ]
    colonnes_retraites_uniques = [
        "retirement_pay", "Former_dep",
        "Former_emp_contract", "distance_former_km"
    ]
    colonnes_num_partagees = [col for col in colonnes_num if col not in (colonnes_actifs_uniques + colonnes_retraites_uniques)]
    colonnes_cat_partagees = [col for col in colonnes_cat if col not in (colonnes_actifs_uniques + colonnes_retraites_uniques)]
    
    # Pipeline pour les variables numériques partagées
    pipeline_num_partagees = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Pipeline pour les variables numériques spécifiques aux "Actifs"
    pipeline_num_actifs = ColumnTransformer([
        ("knn_remu", pipeline_knn(), ["Remuneration"]),
        ("knn_hours", pipeline_knn(), ["working_hours"]),
        ("knn_dist", pipeline_knn(), ["distance_job_km"]),
        ("autres", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler())
        ]), [col for col in colonnes_actifs_uniques 
             if col not in ["Remuneration", "working_hours", "distance_job_km"] and col in colonnes_num])
    ])
    
    # Pipeline pour les variables numériques spécifiques aux "Retraités"
    pipeline_num_retraites = ColumnTransformer([
        ("knn_ret_pay", pipeline_knn(), ["retirement_pay"]),
        ("knn_ret_dist", pipeline_knn(), ["distance_former_km"]),
        ("autres", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler())
        ]), [col for col in colonnes_retraites_uniques 
             if col not in ["retirement_pay", "distance_former_km"] and col in colonnes_num])
    ])
    
    # Pipeline pour les variables catégorielles
    convert_to_str = FunctionTransformer(to_str, feature_names_out="one-to-one")
    
    pipeline_cat_partagees = Pipeline([
        ("to_str", convert_to_str),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Non renseigné")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pipeline_cat_actifs = Pipeline([
        ("to_str", convert_to_str),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Inconnu")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pipeline_cat_retraites = Pipeline([
        ("to_str", convert_to_str),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Inconnu")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Assemblage de la pipeline globale de prétraitement
    preprocessor = ColumnTransformer([
        ("num_partages", pipeline_num_partagees, colonnes_num_partagees),
        ("num_actifs", pipeline_num_actifs, [col for col in colonnes_actifs_uniques if col in colonnes_num]),
        ("num_retraites", pipeline_num_retraites, [col for col in colonnes_retraites_uniques if col in colonnes_num]),
        ("cat_partages", pipeline_cat_partagees, colonnes_cat_partagees),
        ("cat_actifs", pipeline_cat_actifs, [col for col in colonnes_actifs_uniques if col in colonnes_cat]),
        ("cat_retraites", pipeline_cat_retraites, [col for col in colonnes_retraites_uniques if col in colonnes_cat]),
        ("flag_statut", "passthrough", ["Statut_encoded"]),
        ("flags_missing", "passthrough", [
            "Remuneration_missing", "distance_job_km_missing",
            "retirement_pay_missing", "distance_former_km_missing"
        ])
    ])
    return preprocessor