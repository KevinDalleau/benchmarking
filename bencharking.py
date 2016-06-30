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
from sklearn.metrics.cluster import homogeneity_completeness_v_measure

numpy.set_printoptions(threshold=numpy.nan)

data = numpy.loadtxt("./data/soybean.csv", delimiter=",",dtype="float",usecols=range(1,36),skiprows=1)
labels = numpy.loadtxt("./data/soybean.csv", delimiter=",",dtype="string",usecols=(0,),skiprows=1)
#labels = [0 for i in range(len(data))]
data_agg = numpy.loadtxt("./data/soybeanagg.csv", delimiter=",",dtype="float")
data_agg2 = numpy.loadtxt("./data/soyagg2.csv", delimiter=",",dtype="float")
data_agg4 = numpy.loadtxt("./data/soyagg4_leaf_seed.csv", delimiter=",",dtype="float")
data_agg4_w = numpy.loadtxt("./data/soyagg4_ls_weather.csv", delimiter=",",dtype="float")
data_agg4_ws = numpy.loadtxt("./data/soyagg4_lsw_season.csv", delimiter=",",dtype="float")
data_agg4_just_season = numpy.loadtxt("./data/soyagg4_just_season.csv", delimiter=",",dtype="float")
data_agg4_just_leaf = numpy.loadtxt("./data/soyagg4_just_leaf.csv", delimiter=",",dtype="float")
data_agg4_just_seed = numpy.loadtxt("./data/soyagg4_just_seed.csv", delimiter=",",dtype="float")
data_agg4_just_weather = numpy.loadtxt("./data/soyagg4_just_weather.csv", delimiter=",",dtype="float")


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
cluster_agg_ap2 = clusterer_agg_ap.fit_predict(data_agg2)
cluster_agg_ap4 = clusterer_agg_ap.fit_predict(data_agg4)
cluster_agg_ap4_w = clusterer_agg_ap.fit_predict(data_agg4_w)
cluster_agg_ap4_ws = clusterer_agg_ap.fit_predict(data_agg4_ws)
cluster_agg_ap4_just_season = clusterer_agg_ap.fit_predict(data_agg4_just_season)
cluster_agg_ap4_just_leaf = clusterer_agg_ap.fit_predict(data_agg4_just_leaf)
cluster_agg_ap4_just_seed = clusterer_agg_ap.fit_predict(data_agg4_just_seed)
cluster_agg_ap4_just_weather = clusterer_agg_ap.fit_predict(data_agg4_just_weather)


mutual_info_score = adjusted_mutual_info_score(labels,cluster_ap)
mutual_info_score_agg = adjusted_mutual_info_score(labels,cluster_agg_ap)

v_score = homogeneity_completeness_v_measure(labels,cluster_ap)
v_score_agg2 = homogeneity_completeness_v_measure(labels,cluster_agg_ap2)
v_score_agg4 = homogeneity_completeness_v_measure(labels,cluster_agg_ap4)
v_score_agg4_w = homogeneity_completeness_v_measure(labels,cluster_agg_ap4_w)
v_score_agg4_ws = homogeneity_completeness_v_measure(labels,cluster_agg_ap4_ws)
v_score_agg4_just_season = homogeneity_completeness_v_measure(labels,cluster_agg_ap4_just_season)
v_score_agg4_just_leaf = homogeneity_completeness_v_measure(labels,cluster_agg_ap4_just_leaf)
v_score_agg4_just_seed = homogeneity_completeness_v_measure(labels,cluster_agg_ap4_just_seed)
v_score_agg4_just_weather = homogeneity_completeness_v_measure(labels,cluster_agg_ap4_just_weather)


print(v_score)
print(v_score_agg2)
print(v_score_agg4_just_leaf)
print(v_score_agg4_just_seed)
print(v_score_agg4) # Leaf and seed
print(v_score_agg4_just_weather)
print(v_score_agg4_w)
print(v_score_agg4_just_season)
print(v_score_agg4_ws)
