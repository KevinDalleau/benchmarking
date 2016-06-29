# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 22:07:23 2016

@author: kevin
"""

import sklearn
import sklearn.cluster as cluster
from sklearn.metrics import f1_score
import numpy
import csv
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score

numpy.set_printoptions(threshold=numpy.nan)

data = numpy.loadtxt("/home/kevin/Desktop/soybean.csv", delimiter=",",dtype="float",usecols=range(1,36),skiprows=1)
labels = numpy.loadtxt("/home/kevin/Desktop/soybean.csv", delimiter=",",dtype="string",usecols=(0,),skiprows=1)
#labels = [0 for i in range(len(data))]
data_agg = numpy.loadtxt("/home/kevin/Desktop/soybeanagg.csv", delimiter=",",dtype="float")
clusterer_sc = cluster.SpectralClustering(n_clusters=15)
clusterer_agg_sc = cluster.SpectralClustering(n_clusters=15,affinity="precomputed")
cluster_sc = clusterer_sc.fit_predict(data)
cluster_agg_sc = clusterer_agg_sc.fit_predict(data_agg)

clusterer_ac = cluster.AgglomerativeClustering(n_clusters=15,linkage="complete", affinity="manhattan")
clusterer_agg_ac = cluster.AgglomerativeClustering(n_clusters=15, linkage="complete",affinity="precomputed")
cluster_ac = clusterer_ac.fit_predict(data)
cluster_agg_ac = clusterer_agg_ac.fit_predict(data_agg)

clusterer_ap = cluster.AffinityPropagation()
clusterer_agg_ap = cluster.AffinityPropagation(affinity="precomputed")
cluster_ap = clusterer_ap.fit_predict(data)
cluster_agg_ap = clusterer_agg_ap.fit_predict(data_agg)

mutual_info_score = adjusted_mutual_info_score(labels,cluster_ap)
mutual_info_score_agg = adjusted_mutual_info_score(labels,cluster_agg_ap)

v_score = v_measure_score(labels,cluster_sc)
v_score_agg = v_measure_score(labels,cluster_agg_sc)

print(v_score)
print(v_score_agg)