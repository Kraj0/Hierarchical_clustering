from sklearn.datasets import load_iris
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Wczytanie danych Iris
iris = load_iris()
data = iris.data

def manual_hierarchical_clustering(data, n_clusters, linkage_method):
    distance_matrix = squareform(pdist(data, metric="euclidean"))
    clusters = [[i] for i in range(len(data))]

    while len(clusters) > n_clusters:
        min_dist = float("inf")
        to_merge = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = linkage_distance(distance_matrix, clusters[i], clusters[j], linkage_method)
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)

        i, j = to_merge
        clusters[i].extend(clusters[j])
        del clusters[j]

    return clusters


def linkage_distance(distance_matrix, cluster_a, cluster_b, method):
    indices_a = np.array(cluster_a)
    indices_b = np.array(cluster_b)

    if method == "single":
        return np.min(distance_matrix[np.ix_(indices_a, indices_b)])
    elif method == "complete":
        return np.max(distance_matrix[np.ix_(indices_a, indices_b)])
    elif method == "average":
        return np.mean(distance_matrix[np.ix_(indices_a, indices_b)])
    else:
        raise ValueError("Nieobsługiwana metoda łączenia")


def plot_manual_dendrogram(data, linkage_method):
    condensed_distance_matrix = squareform(pdist(data, metric="euclidean"))
    plt.figure(figsize=(10, 7), dpi=150)
    dendrogram(linkage(condensed_distance_matrix, method=linkage_method))
    plt.title(f"Dendrogram - Manual ({linkage_method.capitalize()})")
    plt.xlabel("Indeks próbki")
    plt.ylabel("Odległość")
    plt.show()


def silhouette_analysis_manual(data, linkage_method):
    silhouette_scores = []
    cluster_range = range(2, 6)
    global manual_labels
    for n_clusters in cluster_range:
        manual_clusters = manual_hierarchical_clustering(data, n_clusters, linkage_method)
        manual_labels = np.zeros(len(data), dtype=int)
        for cluster_id, cluster in enumerate(manual_clusters):
            for point in cluster:
                manual_labels[point] = cluster_id
        score = silhouette_score(data, manual_labels)
        silhouette_scores.append((n_clusters, score))

    plt.figure(figsize=(8, 6))
    scores_df = pd.DataFrame(silhouette_scores, columns=["Liczba klastrów", "Współczynnik silhouette"])
    plt.plot(scores_df["Liczba klastrów"], scores_df["Współczynnik silhouette"], marker="o", linestyle="-",
             label="Manual")
    plt.title(f"Współczynnik silhouette - Manual ({linkage_method.capitalize()})")
    plt.xlabel("Liczba klastrów")
    plt.ylabel("Współczynnik silhouette")
    plt.grid()
    plt.legend()
    plt.show()


def plot_library_dendrogram(data, linkage_method):
    linked = linkage(data, method=linkage_method)
    plt.figure(figsize=(10, 7), dpi=150)
    dendrogram(linked)
    plt.title(f"Dendrogram - Library ({linkage_method.capitalize()})")
    plt.xlabel("Indeks próbki")
    plt.ylabel("Odległość")
    plt.show()


def silhouette_analysis_library(data, linkage_method):
    silhouette_scores = []
    cluster_range = range(2, 6)
    global library_labels
    for n_clusters in cluster_range:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        library_labels = clustering.fit_predict(data)
        score = silhouette_score(data, library_labels)
        silhouette_scores.append((n_clusters, score))

    #Wykres
    plt.figure(figsize=(8, 6))
    scores_df = pd.DataFrame(silhouette_scores, columns=["Liczba klastrów", "Współczynnik silhouette"])
    plt.plot(scores_df["Liczba klastrów"], scores_df["Współczynnik silhouette"], marker="o", linestyle="-",
             label="Library")
    plt.title(f"Współczynnik silhouette - Library ({linkage_method.capitalize()})")
    plt.xlabel("Liczba klastrów")
    plt.ylabel("Współczynnik silhouette")
    plt.grid()
    plt.legend()
    plt.show()


def run_analysis_for_method(data, linkage_method):
    print(f"\n=== Analiza dla metody: {linkage_method.capitalize()} ===")

    manual_clusters = manual_hierarchical_clustering(data, 3, linkage_method)
    plot_manual_dendrogram(data, linkage_method)
    silhouette_analysis_manual(data, linkage_method)

    plot_library_dendrogram(data, linkage_method)
    silhouette_analysis_library(data, linkage_method)

    #Tworzenie tabeli zgodności
    contingency_table = pd.crosstab(pd.Series(manual_labels, name="Manual"),
                                    pd.Series(library_labels, name="Library"))


    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Tabela zgodności klastrów ({linkage_method.capitalize()})")
    plt.xlabel("Library Labels")
    plt.ylabel("Manual Labels")
    plt.show()


    plt.figure(figsize=(10, 5))

    #Labele
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=manual_labels, cmap="viridis", s=50)
    plt.title(f"Manual Labels ({linkage_method.capitalize()})")
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Sepal width (cm)")

    #Labele
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c=library_labels, cmap="viridis", s=50)
    plt.title(f"Library Labels ({linkage_method.capitalize()})")
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Sepal width (cm)")

    plt.tight_layout()
    plt.show()


#Wywołanie analiz dla metod single, complete i average
methods = ["single", "complete", "average"]
for method in methods:
    run_analysis_for_method(data, method)
