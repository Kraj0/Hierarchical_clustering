import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from clustering import (
    manual_hierarchical_clustering,
    linkage_distance,
    silhouette_analysis_manual,
    silhouette_analysis_library,
    plot_manual_dendrogram,
    plot_library_dendrogram,
)

class TestHierarchicalClustering(unittest.TestCase):
    def setUp(self):
        # Przygotowanie danych Iris
        iris = load_iris()
        self.data = iris.data
        self.distance_matrix = squareform(pdist(self.data, metric="euclidean"))

    def test_manual_clustering_single(self):
        clusters = manual_hierarchical_clustering(self.data, 3, "single")
        self.assertEqual(len(clusters), 3)
        all_points = [point for cluster in clusters for point in cluster]
        self.assertCountEqual(all_points, range(len(self.data)))

    def test_manual_clustering_complete(self):
        clusters = manual_hierarchical_clustering(self.data, 3, "complete")
        self.assertEqual(len(clusters), 3)
        all_points = [point for cluster in clusters for point in cluster]
        self.assertCountEqual(all_points, range(len(self.data)))

    def test_manual_clustering_average(self):
        clusters = manual_hierarchical_clustering(self.data, 3, "average")
        self.assertEqual(len(clusters), 3)
        all_points = [point for cluster in clusters for point in cluster]
        self.assertCountEqual(all_points, range(len(self.data)))

    def test_linkage_distance(self):
        cluster_a = [0, 1]
        cluster_b = [2, 3]
        single = linkage_distance(self.distance_matrix, cluster_a, cluster_b, "single")
        complete = linkage_distance(self.distance_matrix, cluster_a, cluster_b, "complete")
        average = linkage_distance(self.distance_matrix, cluster_a, cluster_b, "average")

        self.assertAlmostEqual(single, np.min(self.distance_matrix[np.ix_(cluster_a, cluster_b)]))
        self.assertAlmostEqual(complete, np.max(self.distance_matrix[np.ix_(cluster_a, cluster_b)]))
        self.assertAlmostEqual(average, np.mean(self.distance_matrix[np.ix_(cluster_a, cluster_b)]))

    def test_silhouette_analysis_manual(self):
        try:
            silhouette_analysis_manual(self.data, "average")
        except Exception as e:
            self.fail(f"Silhouette analysis for manual method failed with error: {e}")

    def test_silhouette_analysis_library(self):
        try:
            silhouette_analysis_library(self.data, "average")
        except Exception as e:
            self.fail(f"Silhouette analysis for library method failed with error: {e}")

    def test_dendrogram_plot_manual(self):
        try:
            plot_manual_dendrogram(self.data, "average")
        except Exception as e:
            self.fail(f"Manual dendrogram plotting failed with error: {e}")

    def test_dendrogram_plot_library(self):
        try:
            plot_library_dendrogram(self.data, "average")
        except Exception as e:
            self.fail(f"Library dendrogram plotting failed with error: {e}")

if __name__ == "__main__":
    unittest.main()