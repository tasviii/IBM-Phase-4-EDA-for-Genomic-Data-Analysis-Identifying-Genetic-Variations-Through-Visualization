import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import chi2_contingency, f_oneway
import streamlit as st

# File paths (update these with actual file paths)
file_paths = {
    "Father": "/content/Father Genome.csv",
    "Mother": "/content/Mother Genome.csv",
    "Child 1": "/content/Child 1 Genome.csv",
    "Child 2": "/content/Child 2 Genome.csv",
    "Child 3": "/content/Child 3 Genome.csv",
}

# Load datasets with dtype specification to handle mixed types
datasets = {name: pd.read_csv(path, dtype=str) for name, path in file_paths.items()}

# Keep only SNPs and genotypes, dropping metadata columns
for name, df in datasets.items():
    df.drop(columns=["chromosome", "position"], inplace=True, errors="ignore")
    df.rename(columns={"genotype": name}, inplace=True)

# Merge all datasets on rsid using an INNER JOIN to avoid NaNs
merged_df = datasets["Father"]
for name in ["Mother", "Child 1", "Child 2", "Child 3"]:
    merged_df = merged_df.merge(datasets[name], on="# rsid", how="inner")  # Changed to inner join

# Drop SNP ID column
df_numeric = merged_df.drop(columns=["# rsid"], errors="ignore")

# Encode genotypes numerically. Handle potential errors during encoding.
encoder = LabelEncoder()
df_encoded = df_numeric.apply(lambda col: encoder.fit_transform(col.astype(str)))

# Standardize the dataset
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)


# ======== Incremental PCA ========
ipca = IncrementalPCA(n_components=3, batch_size=500)
pca_result = ipca.fit_transform(df_scaled)

# Convert to DataFrame
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2", "PC3"])

# Correctly assign the "Individual" column:
individual_labels = list(file_paths.keys())
num_individuals = len(individual_labels)
num_rows_pca = pca_df.shape[0]

# Calculate the number of full cycles of individuals
full_cycles = num_rows_pca // num_individuals

# Create the base set of labels (repeated full cycles)
individual_column = np.repeat(individual_labels, full_cycles)

# Add any remaining labels (for the partial cycle)
remaining_rows = num_rows_pca % num_individuals
if remaining_rows > 0:
    individual_column = np.concatenate([individual_column, individual_labels[:remaining_rows]])

pca_df["Individual"] = individual_column



# ======== t-SNE Implementation (Sampling for Efficiency) ========
sample_size = min(5000, df_scaled.shape[0]) if df_scaled.shape[0] > 1 else df_scaled.shape[0]
df_sample = df_scaled[np.random.choice(df_scaled.shape[0], sample_size, replace=False)]

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
tsne_result = tsne.fit_transform(df_sample)

# Convert to DataFrame
tsne_df = pd.DataFrame(tsne_result, columns=["TSNE1", "TSNE2"])
tsne_df["Individual"] = np.tile(list(file_paths.keys()), len(tsne_df) // len(file_paths))



# ======== Statistical Tests for Significant Genetic Markers ========
chi_square_results = {}
anova_results = {}

for snp in df_encoded.columns:
    try:
        # Chi-Square Test for each SNP
        contingency_table = pd.crosstab(merged_df[snp], merged_df["Father"])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        chi_square_results[snp] = p

        # ANOVA Test
        groups = [df_encoded[snp][df_encoded.index % len(file_paths) == i] for i in range(len(file_paths))]
        f_stat, p_anova = f_oneway(*groups)
        anova_results[snp] = p_anova
    except Exception as e:
        print(f"Skipping SNP {snp} due to error: {e}")

# Filter significant SNPs
significant_snps_chi2 = {snp: p for snp, p in chi_square_results.items() if p < 0.05}
significant_snps_anova = {snp: p for snp, p in anova_results.items() if p < 0.05}


# Streamlit app
st.title("Genomic Data Analysis")

# ... (Data loading and preprocessing - same as before)


# ======== Incremental PCA ========
st.subheader("3D PCA of Genomic Data")
fig_pca = plt.figure(figsize=(8, 6))
ax_pca = fig_pca.add_subplot(111, projection="3d")

colors = sns.color_palette("husl", len(file_paths))
for i, individual in enumerate(file_paths.keys()):
    subset = pca_df[pca_df["Individual"] == individual]
    ax_pca.scatter(subset["PC1"], subset["PC2"], subset["PC3"], label=individual, color=colors[i], alpha=0.6)

ax_pca.set_xlabel("PC1")
ax_pca.set_ylabel("PC2")
ax_pca.set_zlabel("PC3")
ax_pca.set_title("3D PCA of Genomic Data")  # Setting title for the axes object
ax_pca.legend()
st.pyplot(fig_pca)  # Use st.pyplot to display the Matplotlib figure



# ======== t-SNE Implementation (Sampling for Efficiency) ========
st.subheader("t-SNE Visualization of Genomic Data")
fig_tsne = plt.figure(figsize=(8, 6))
ax_tsne = fig_tsne.add_subplot(111)  # Create a subplot for t-SNE

for i, individual in enumerate(file_paths.keys()):
    subset = tsne_df[tsne_df["Individual"] == individual]
    ax_tsne.scatter(subset["TSNE1"], subset["TSNE2"], label=individual, color=colors[i], alpha=0.6)

ax_tsne.set_xlabel("t-SNE1")
ax_tsne.set_ylabel("t-SNE2")
ax_tsne.set_title("t-SNE Visualization of Genomic Data")  # Setting title for the axes object
ax_tsne.legend()
st.pyplot(fig_tsne)  # Use st.pyplot to display the Matplotlib figure



# ======== Statistical Tests for Significant Genetic Markers ========
st.subheader("Significant SNPs (Chi-Square Test)")
st.write(significant_snps_chi2)

st.subheader("Significant SNPs (ANOVA Test)")
st.write(significant_snps_anova)
