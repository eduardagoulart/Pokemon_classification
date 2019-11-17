'''
Adapted from: https://towardsdatascience.com/kmeans-clustering-for-classification-74b992405d0a
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


class Clust():

    def _load_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.7, random_state=42)
        self.n_clusters = 0

    def __init__(self, df):
        against = [
        "against_bug",
        "against_dark",
        "against_dragon",
        "against_electric",
        "against_fairy",
        "against_fight",
        "against_fire",
        "against_flying",
        "against_ghost",
        "against_grass",
        "against_ground",
        "against_ice",
        "against_normal",
        "against_poison",
        "against_psychic",
        "against_rock",
        "against_steel",
        "against_water",
        ]
        X = df[against]
        y = df['type1']
        self._load_data(X, y)

    def classify(self, outputFile, model=RandomForestClassifier(n_estimators = 100)):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        outputFile.write(str(accuracy_score(self.y_test, y_pred))+", " + str(self.n_clusters) + ", " + self.mode + "\n")

    def Kmeans(self, n_clusters, output='add'):
        self.mode = output
        self.n_clusters = n_clusters#len(np.unique(self.y_train))
        clf = KMeans(n_clusters = self.n_clusters, random_state=42)
        clf.fit(self.X_train)
        y_labels_train = clf.labels_
        y_labels_test = clf.predict(self.X_test)
        if output == 'add':
            self.X_train['km_clust'] = y_labels_train
            self.X_test['km_clust'] = y_labels_test
        elif output == 'replace':
            self.X_train = y_labels_train[:, np.newaxis]
            self.X_test = y_labels_test[:, np.newaxis]
        else:
            raise ValueError('output should be either add or replace')
        return self