# Databricks notebook source
# MAGIC %pip install scikit-learn==1.2.2

# COMMAND ----------

# MAGIC %pip install scikit-learn-extra

# COMMAND ----------

import sklearn

print(sklearn.__version__)

# COMMAND ----------

import glob
import pyspark.sql.functions as f

feature_dir = "/mnt/dev/customer_segmentation/imx/joyce_beauty/features"
files = glob.glob("/dbfs" + feature_dir + "/*.parquet")

features_df = None
for path in files:
    if "demographic" in path or "transactional" in path:
        continue
    if "tagging.parquet" not in path:
        df = spark.read.parquet(path.replace("/dbfs", ""))
        if features_df is None:
            features_df = df
        else:
            features_df = features_df.join(df, on='vip_main_no', how='inner')
    else:
        df = spark.read.parquet(path.replace("/dbfs", ""))
        features_df = features_df.join(df, on='vip_main_no', how='left')

# COMMAND ----------

# remove outlier 
features_df = features_df.filter(~features_df["vip_main_no"].isin(["JBH21B000457", "JBH21B002071"]))

# COMMAND ----------

features_df = features_df.fillna(0)

# remove features with 85% zero
zero_percentage_threshold = 0.85
zero_features = []

total_rows = features_df.count()
for column_name in features_df.columns:
    zero_count = features_df.filter(f.col(column_name) == 0).count()
    zero_percentage = zero_count / total_rows
    
    if zero_percentage >= zero_percentage_threshold:
        zero_features.append(column_name)

print(zero_features, len(zero_features))

# COMMAND ----------

# manual select feature
features_to_keep = [col for col in features_df.columns if col not in zero_features and "brand" not in col and "SET" not in col and "Set" not in col]
features_df_filtered = features_df.select(features_to_keep)

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

feature_cols = [c for c in features_df_filtered.columns if c != "vip_main_no"]
feature_cols, len(feature_cols)

# COMMAND ----------

all_vip = features_df_filtered.select("vip_main_no").toPandas().values.reshape(1, -1)

# COMMAND ----------

all_vip

# COMMAND ----------

pandas_df = features_df_filtered.select(feature_cols).toPandas()
features_array = pandas_df.values
features_array.shape

# COMMAND ----------

# standardization
scaler = StandardScaler()
standardized_df = scaler.fit_transform(features_array)
standardized_df = np.nan_to_num(standardized_df)

# COMMAND ----------

# FactorAnalysis
n_components = 25

fa = FactorAnalysis(n_components=n_components)
fa.fit(standardized_df)

factor_loadings = fa.components_
selected_features = np.abs(factor_loadings).sum(axis=0).argsort()[:n_components]

# COMMAND ----------

np.array(feature_cols)[selected_features]

# COMMAND ----------

factor_table = pd.DataFrame(factor_loadings.T, index=np.array(feature_cols))
factor_table

# COMMAND ----------

standardized_df.shape

# COMMAND ----------

features_embed = standardized_df[:,selected_features]
features_embed.shape

# COMMAND ----------

from sklearn.cluster import KMeans

# kmeans
wcss = [] # Within-Cluster Sum of Square -> variability of the observations within each cluster
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, n_init=10)
    kmeans.fit(features_embed)
    wcss.append(kmeans.inertia_)

# COMMAND ----------

# Elbow-method
plt.figure(figsize=(10, 8))
plt.plot(range(1, 21), wcss, marker="o", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# COMMAND ----------

# k-mean final
kmeans = KMeans(n_clusters=5, n_init="auto")
kmeans.fit(features_embed)

# COMMAND ----------

# visualize cluster
from sklearn.manifold import TSNE

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(features_embed)

# Create scatter plot with colored clusters
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans.labels_)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('K-means Clustering Result (t-SNE Visualization)')
plt.show()

# COMMAND ----------

for i in range(5):
    print(np.sum(kmeans.labels_ == i))

# COMMAND ----------

np.unique(kmeans.labels_, return_counts=True)

# COMMAND ----------

import pandas as pd
import numpy as np

result_df = pd.DataFrame(np.concatenate((all_vip.reshape(-1, 1), kmeans.labels_.reshape(-1, 1)), axis=1), columns=["vip_main_no", "persona"])
spark.createDataFrame(result_df).write.parquet("/mnt/dev/customer_segmentation/imx/joyce_beauty/model/clustering_result_kmeans_iter1.parquet")

# COMMAND ----------

# k-mean final
kmeans = KMeans(n_clusters=4, n_init=10)
kmeans.fit(features_embed)

# Create scatter plot with colored clusters
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans.labels_)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('K-means Clustering Result (t-SNE Visualization)')
plt.show()

# COMMAND ----------

np.unique(kmeans.labels_, return_counts=True)

# COMMAND ----------

result_df = pd.DataFrame(np.concatenate((all_vip.reshape(-1, 1), kmeans.labels_.reshape(-1, 1)), axis=1), columns=["vip_main_no", "persona"])

# COMMAND ----------

features_subset = features_embed[kmeans.labels_ == 1]
vip_subset = all_vip[0][kmeans.labels_ == 1]

# COMMAND ----------

vip_subset.shape

# COMMAND ----------

# break down the larger cluster
kmeans_small = KMeans(n_clusters=2, n_init=10)
kmeans_small.fit(features_subset)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(features_subset)

# Create scatter plot with colored clusters
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_small.labels_)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('K-means Clustering Result (t-SNE Visualization)')
plt.show()

# COMMAND ----------

np.unique(kmeans_small.labels_, return_counts=True)

# COMMAND ----------

result_subset_df = pd.DataFrame(np.concatenate((vip_subset.reshape(-1, 1), kmeans_small.labels_.reshape(-1, 1)), axis=1), columns=["vip_main_no", "persona"])

# COMMAND ----------

result_df = result_df[result_df["persona"] != 1]

# COMMAND ----------

len(result_df)

# COMMAND ----------

result_subset_df["persona"] = result_subset_df["persona"].apply(lambda x: 1 if x == 0 else 5)

# COMMAND ----------

len(result_subset_df)

# COMMAND ----------

final_result = pd.concat([result_df, result_subset_df]).reset_index(drop=True)

# COMMAND ----------

final_result

# COMMAND ----------

spark.createDataFrame(final_result).write.parquet("/mnt/dev/customer_segmentation/imx/joyce_beauty/model/clustering_result.parquet")

# COMMAND ----------

# save model
import os
import joblib

model_dir = "/dbfs/mnt/dev/customer_segmentation/imx/joyce_beauty/model/"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(kmeans, os.path.join(model_dir, "kmeans_model.pkl"))

# COMMAND ----------

vips = features_df[["vip_main_no"]].toPandas().values
vips_persona = np.concatenate((vips.reshape(-1, 1), kmeans.labels_.reshape(-1, 1)), axis=1)
np.save(os.path.join(model_dir, "vips_persona.npy"), vips_persona)

# COMMAND ----------

# kmeans with contraint 

# COMMAND ----------

pip install k-means-constrained

# COMMAND ----------

from k_means_constrained import KMeansConstrained

wcss = [] # Within-Cluster Sum of Square -> variability of the observations within each cluster
for i in range(1, 21):
    kmeans_c = KMeansConstrained(n_clusters=i, size_min=1000)
    kmeans_c.fit(features_embed)
    wcss.append(kmeans_c.inertia_)



# COMMAND ----------

# Elbow-method
plt.figure(figsize=(10, 8))
plt.plot(range(1, 15), wcss, marker="o", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# COMMAND ----------

kmeans_c = KMeansConstrained(n_clusters=4, size_min=2532, size_max=5839)
kmeans_c.fit(features_embed)

# COMMAND ----------

kmeans_c.labels_

# COMMAND ----------

kmeans_c.inertia_

# COMMAND ----------

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_c.labels_)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('K-means Clustering Result (t-SNE Visualization)')
plt.show()

# COMMAND ----------

np.unique(kmeans_c.labels_, return_counts=True)

# COMMAND ----------

len(kmeans_c.labels_)

# COMMAND ----------

len(vips)

# COMMAND ----------

vips = features_df[["vip_main_no"]].toPandas().values
vips_persona = np.concatenate((vips.reshape(-1, 1), kmeans_c.labels_.reshape(-1, 1)), axis=1)
np.save(os.path.join(model_dir, "vips_persona.npy"), vips_persona)

# COMMAND ----------

for i in range(4):
    print(np.sum(kmeans_c.labels_ == i))

# COMMAND ----------

kmeans_c = KMeansConstrained(n_clusters=5, size_min=1000, size_max=None)
kmeans_c.fit(features_embed)

# COMMAND ----------

# KMedoids
from sklearn_extra.cluster import KMedoids

sum_of_distances = [] # sum of distances of samples to their closest cluster center.
for i in range(1, 21):
    kmedoids = KMedoids(n_clusters=i, random_state=42)
    kmedoids.fit(features_embed)
    sum_of_distances.append(kmedoids.inertia_)

# COMMAND ----------

# Elbow-method
plt.figure(figsize=(10, 8))
plt.plot(range(1, 21), sum_of_distances, marker="o", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of distances of samples to their closest cluster center")
plt.show()

# COMMAND ----------

# kmedoids final
kmedoids = KMedoids(n_clusters=4, random_state=42)
kmedoids.fit(features_embed)

# COMMAND ----------

np.unique(kmedoids.labels_, return_counts=True)

# COMMAND ----------

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(features_embed)

# Create scatter plot with colored clusters
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmedoids.labels_)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('K-Medoids Clustering Result (t-SNE Visualization)')
plt.show()

# COMMAND ----------

# DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust the hyperparameters as needed
dbscan.fit(features_embed)

# COMMAND ----------

# Create scatter plot with colored clusters
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=dbscan.labels_)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('K-means Clustering Result (t-SNE Visualization)')
plt.show()

# COMMAND ----------

# hierarchical clustering
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=5, compute_distances=True)
model.fit(features_embed)

# COMMAND ----------

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=model.labels_)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('K-means Clustering Result (t-SNE Visualization)')
plt.show()

# COMMAND ----------

from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plot_dendrogram(model, truncate_mode='level', p=6)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# COMMAND ----------

plot_dendrogram(model, truncate_mode='level', p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# COMMAND ----------

plot_dendrogram(model, truncate_mode='level', p=4)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# COMMAND ----------


