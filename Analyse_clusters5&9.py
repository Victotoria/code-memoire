import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import (MinMaxScaler)

# === PARAMÈTRES ===
EXCEL_PATH = ""
CLUSTERS_TO_ANALYZE = [5, 9]
EXCLUDE_CENTRE = True  # Passe à False si tu veux garder les bassins du centre

output_dir = ""
os.makedirs(output_dir, exist_ok=True)

# === CHARGEMENT DES DONNÉES ===
df = pd.read_excel(EXCEL_PATH)

# === FILTRAGE CLUSTERS & RIVE ===
df = df[df["cluster"].isin(CLUSTERS_TO_ANALYZE)]
if EXCLUDE_CENTRE:
    df = df[df["rive"].isin(["Est", "Ouest"])]
    
# === VARIABLES CONTINUES ===
variables_continues = [
    "AREA", "PERIMETER", "Circularit", "den_draina", "alt_mean", 
    "alt_min", "alt_max", "HI", "LENGTH_sum", "slope_mean", 
    "slope_min", "slope_max", "aspect_med", "TWI_mean", "TWI_stdev",
    "TWI_min", "TWI_max", "SPI_mean", "SPI_stdev", "SPI_min", "SPI_max",
    "rugo_mean", "rugo_min", "rugo_max", "nbr_knickp", "susceptibi", 
    "KSN_mean", "KSN_median", "KSN_min", "KSN_max", "nbr_shallo", 
    "nbr_profon", "den_shallo", "den_profon", "def_2014-2", 
    "def_2014_pct", "pct_bati", "pct_foret", "pct_vegetati", "pct_cult",
    "forest_59_pct", "forest_20_pct", "forest_74_pct",
    "forest_change_59_74", "forest_change_74_20", "forest_change_59_20", "barrage present"
]

# === FILTRAGE DES CLUSTERS ===
df = df[df["cluster"].isin(CLUSTERS_TO_ANALYZE)]
if EXCLUDE_CENTRE:
    df = df[df["rive"].isin(["Est", "Ouest"])]

# === PROPORTION EST / OUEST PAR CLUSTER ===
pivot_rive = df.pivot_table(index="cluster", columns="rive", values="fid", aggfunc="count").fillna(0)
pivot_rive["Total"] = pivot_rive.sum(axis=1)
pivot_rive["% Est"] = (pivot_rive["Est"] / pivot_rive["Total"] * 100).round(1)
pivot_rive["% Ouest"] = (pivot_rive["Ouest"] / pivot_rive["Total"] * 100).round(1)
pivot_rive.to_csv(os.path.join(output_dir, "repartition_est_ouest.csv"))

# === APPARITION DE DELTAS / CHANGEMENT SINUOSITÉ PAR CLUSTER ===
delta_summary = df.groupby("cluster")["delta_anytime"].mean().round(2)
sin_summary = df.groupby("cluster")["sin_change_category"].value_counts(normalize=True).unstack().round(2)
delta_summary.to_csv(os.path.join(output_dir, "delta_par_cluster.csv"))
sin_summary.to_csv(os.path.join(output_dir, "sinuosite_par_cluster.csv"))

# === NORMALISATION DES VARIABLES CONTINUES ===
scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[variables_continues] = scaler.fit_transform(df[variables_continues])

# === MOYENNES NORMALISÉES PAR CLUSTER ET RIVE ===
def mean_by_cluster_rive(df_scaled, cluster_num, rive):
    subset = df_scaled[(df_scaled["cluster"] == cluster_num) & (df_scaled["rive"] == rive)]
    return subset[variables_continues].mean().round(3)

res_cluster_5_est = mean_by_cluster_rive(df_norm, 5, "Est")
res_cluster_5_ouest = mean_by_cluster_rive(df_norm, 5, "Ouest")
res_cluster_9_est = mean_by_cluster_rive(df_norm, 9, "Est")
res_cluster_9_ouest = mean_by_cluster_rive(df_norm, 9, "Ouest")

summary_factors_norm = pd.DataFrame({
    "Cluster 5 - Est": res_cluster_5_est,
    "Cluster 5 - Ouest": res_cluster_5_ouest,
    "Cluster 9 - Est": res_cluster_9_est,
    "Cluster 9 - Ouest": res_cluster_9_ouest
})
summary_factors_norm.to_csv(os.path.join(output_dir, "facteurs_normalises_par_cluster_et_rive.csv"))

# === VISUALISATION COMPARATIVE NORMALISÉE ===
def plot_comparaison_facteurs_norm(cluster_num):
    est = mean_by_cluster_rive(df_norm, cluster_num, "Est")
    ouest = mean_by_cluster_rive(df_norm, cluster_num, "Ouest")
    comp_df = pd.DataFrame({'Est': est, 'Ouest': ouest})
    comp_df.plot(kind='bar', figsize=(12,6))
    plt.title(f"Comparaison normalisée des facteurs - Cluster {cluster_num}")
    plt.ylabel("Valeur normalisée (0–1)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"facteurs_normalises_comparaison_cluster_{cluster_num}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

plot_comparaison_facteurs_norm(5)
plot_comparaison_facteurs_norm(9)

# === FACTEURS COMMUNS PAR CLUSTER & RIVE ===
def mean_by_cluster_rive(df, cluster_num, rive):
    subset = df[(df["cluster"] == cluster_num) & (df["rive"] == rive)]
    return subset[variables_continues].mean().round(2)

res_cluster_5_est = mean_by_cluster_rive(df, 5, "Est")
res_cluster_5_ouest = mean_by_cluster_rive(df, 5, "Ouest")
res_cluster_9_est = mean_by_cluster_rive(df, 9, "Est")
res_cluster_9_ouest = mean_by_cluster_rive(df, 9, "Ouest")

summary_factors = pd.DataFrame({
    "Cluster 5 - Est": res_cluster_5_est,
    "Cluster 5 - Ouest": res_cluster_5_ouest,
    "Cluster 9 - Est": res_cluster_9_est,
    "Cluster 9 - Ouest": res_cluster_9_ouest
})
summary_factors.to_csv(os.path.join(output_dir, "facteurs_par_cluster_et_rive.csv"))

# === VISUALISATIONS ===
sns.set(style="whitegrid")

# 1. Boxplot des surfaces par cluster
plt.figure(figsize=(8, 5))
sns.boxplot(x="cluster", y="AREA", data=df)
plt.title("Distribution de la surface des bassins (AREA) - Clusters 5 et 9")
plt.savefig(os.path.join(output_dir, "boxplot_area_clusters.png"), dpi=300)
plt.close()

# 2. Barres empilées Est/Ouest
pivot_rive[["Est", "Ouest"]].plot(kind="bar", stacked=True, figsize=(8,5))
plt.title("Répartition Est / Ouest - Clusters 5 et 9")
plt.ylabel("Nombre de bassins")
plt.savefig(os.path.join(output_dir, "repartition_est_ouest.png"), dpi=300)
plt.close()

# 3. Barres delta_anytime
delta_summary.plot(kind="bar", figsize=(6,4))
plt.title("Proportion de bassins avec apparition de delta")
plt.ylabel("Proportion")
plt.savefig(os.path.join(output_dir, "proportion_deltas.png"), dpi=300)
plt.close()

# Répartition des barrages par cluster
barrage_by_cluster = df.groupby("cluster")["barrage present"].mean()
print("Proportion de bassins avec barrage par cluster :")
print(barrage_by_cluster)

# Répartition des barrages par localisation
barrage_by_rive = df[df["Google_loc"] != "Centre"].groupby("Google_loc")["barrage present"].mean()
print("Proportion de bassins avec barrage par rive :")
print(barrage_by_rive)

# Barplot des proportions par rive
plt.figure(figsize=(6, 4))
sns.barplot(x=barrage_by_rive.index, y=barrage_by_rive.values)
plt.title("Proportion de bassins avec barrage par rive")
plt.ylabel("Proportion")
plt.xlabel("Rive")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "barrages_par_rive.png"), dpi=300)
plt.show()

print("✅ Tous les fichiers (CSV + PNG haute résolution) ont été enregistrés dans :", output_dir)

