#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 07:45:46 2018

@author: Mengxi Shen
"""
import pandas as pd
import sklearn.cluster
import matplotlib.pyplot as plt

scores = pd.read_csv("cs1-scores.txt", sep=r"\s+", comment="#")
df = scores/scores.max()
df["Assignment"] = df[["Q1", "Q2", "A1", "A2"]].mean(axis=1)
columns = ["Final", "Assignment"]
kmeans = sklearn.cluster.KMeans(n_clusters=7)
kmeans.fit(df[columns])
centers = pd.DataFrame(kmeans.cluster_centers_, columns=columns)
centers.plot.scatter(columns[0], columns[1], color="red", marker="x", s=150)
plt.scatter(df["Final"], df["Assignment"])
plt.xlabel("final exam score")
plt.ylabel("mean assignment score")
plt.savefig("km7.png", dpi=150)
