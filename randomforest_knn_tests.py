import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from cluster import Clust
from sklearn.neighbors import KNeighborsClassifier

def testCluster(X, y):
    out = open('/cluster/testCluster.csv', "w+")
    out.write("AccuracyNormal, AccuracyCluster, NCluster, TypeKmeans\n")
    for mode in ["add", "replace"]:
        for n_clusters in range(1, 19):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            out.write(str(metrics.accuracy_score(y_test, y_pred)))
            cluster = Clust(df)
            cluster.Kmeans(n_clusters, output=mode).classify(out, model=RandomForestClassifier(n_estimators = 100))

    out.close()

def testTrainSize(X, y):
    outnb = open("test/nb/test_train_size_nest.csv", "w+")
    outrf = open("test/rf/test_train_size.csv", "w+")
    outknn = open("test/knn/test_train_size.csv", "w+")

    outnb.write("size, acc\n")
    outrf.write("size, acc\n")
    outknn.write("size, acc\n")
    for size in range(1, 10):
        size = size/10.0
        acc_nb = 0
        acc_rf = 0
        acc_knn = 0
        for i in range(0, 5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
            clfnb = GaussianNB()
            clfrf = RandomForestClassifier(n_estimators = 96)
            clfknn = KNeighborsClassifier(n_neighbors=50)
            clfnb.fit(X_train,y_train)
            clfrf.fit(X_train, y_train)
            clfknn.fit(X_train, y_train)
            y_pred_nb = clfnb.predict(X_test)
            y_pred_rf = clfrf.predict(X_test)
            y_pred_knn = clfknn.predict(X_test)
            acc_nb = acc_nb + metrics.accuracy_score(y_test, y_pred_nb)
            acc_rf = acc_rf + metrics.accuracy_score(y_test, y_pred_rf)
            acc_knn = acc_knn + metrics.accuracy_score(y_test, y_pred_knn)
        
        acc_nb = acc_nb/5.0
        acc_rf = acc_rf/5.0
        acc_knn = acc_knn/5.0

        outnb.write(str(size) + ",")
        outnb.write(str(acc_nb) + "\n")
        
        outrf.write(str(size) + ",")
        outrf.write(str(acc_rf) + "\n")
    
        outknn.write(str(size) + ",")
        outknn.write(str(acc_knn) + "\n")
    
    outnb.close()
    outrf.close()
    outknn.close()

def testNeighboursTrainSize(X, y):
    out = open("test/knn/test_train_neighbours.csv", "w+")
    out.write("size,neighbours,acc\n")
    for size in range(1, 10):
        testsize = size/10.0
        print("Size: " + str(testsize))
        for n in range(1, 80):
            acc = 0
            for j in range(0, 5):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=42)
                clf_knn = KNeighborsClassifier(n_neighbors=n)
                clf_knn.fit(X_train, y_train)
                y_pred = clf_knn.predict(X_test)
                acc = acc + metrics.accuracy_score(y_test, y_pred)
            acc = acc/5.0
            out.write(str(size) + ",")
            out.write(str(n) + ",")
            out.write(str(acc) + "\n")
    out.close()
    

def testNEstimators(X, y):
    outrf = open("test/rf/test_train_size_nest.csv", "w+")
    outrf.write("size,n_estimator,acc\n")
    acc_rf = 0
    for size in range(1, 10):
        size = size/10.0
        print(size)
        for nest in range(1, 100):
            acc_rf = 0
            for i in range(0, 5):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
                clfrf = RandomForestClassifier(n_estimators = nest)
                clfrf.fit(X_train, y_train)
                y_pred_rf = clfrf.predict(X_test)
                acc_rf = acc_rf + metrics.accuracy_score(y_test, y_pred_rf)

            acc_rf = acc_rf/5.0    
            outrf.write(str(size) + ",")
            outrf.write(str(nest) + ",")
            outrf.write(str(acc_rf) + "\n")
    outrf.close()

def testCrossVal(X, y, out):
    out = open("test/nb/cross_val_size.csv", "w+")
    outrf = open(out, "w+")
    outknn = open("test/knn/cross_val_size.csv", "w+")
    out.write("folds, acc\n")
    outknn.write("folds,acc\n")
    outrf.write("folds,acc\n")
    for size in range(2, 100):
        print("Size: " + str(size))
        acc_nb = 0
        acc_rf = 0
        acc_knn = 0
        for i in range(0, 5):
            clf_nb=GaussianNB()
            clf_knn = KNeighborsClassifier(n_neighbors=50)
            clf_rf = RandomForestClassifier(n_estimators = 100)
            scores_nb = cross_val_score(clf_nb, X, y, cv=size)
            scores_rf = cross_val_score(clf_rf, X, y, cv=size)
            scores_knn = cross_val_score(clf_knn, X, y, cv=size)
            acc_nb = acc_nb + scores_nb.mean()
            acc_rf = acc_rf + scores_rf.mean()
            acc_knn = acc_knn + scores_knn.mean()

        acc_nb = acc_nb/5.0
        acc_rf = acc_rf/5.0
        acc_knn = acc_knn/5.0
        outrf.write(str(size) + ",")
        outrf.write(str(acc_rf) + "\n")
        out.write(str(size) + ",")
        out.write(str(acc_nb) + "\n")
        outknn.write(str(size) + ",")
        outknn.write(str(acc_knn) + "\n")
    out.close()
    outrf.close()
    outknn.close()

def findbest():
    file = open("test/rf/test_train_size_nest_full.csv", "r", newline="")
    reader = csv.reader(file, delimiter=',')
    records = []
    for row in reader:
        if (row[0] == "size"):
            continue
        row[0] = float(row[0])
        row[0] = row[0]
        row[1] = int(row[1])
        row[2] = float(row[2])
        records.append(row)
    
    bests = []
    s = 1
    i = 0
    best_local = [0, 0, 0]
    for r in records:
        if (r[0] == "size"):
            continue
        i = i + 1
        if (r[2] > best_local[2]):
            best_local = r
        if (r[0] == s/10.0):
            s = s + 1
            print(s)
            bests.append(best_local)
            best_local = [0, 0, 0]

    out = open("test/rf/test_train_size_nest_full_best.csv", "w+", newline="")
    out.write('"test_size","n_estimator","acc"\n')
    wr = csv.writer(out, quoting=csv.QUOTE_ALL)
    for r in bests:
        wr.writerow(r)
    out.close()


df = pd.read_csv("pokemon.csv")

#default params
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
    #"attack",
    #"base_egg_steps",
    #"base_happiness",
    #"base_total",
    #"capture_rate",
    #"defense",
    #"experience_growth",
    #"height_m",
    #"hp",
    #"sp_attack",
    #"sp_defense",
    #"speed",
    #"weight_kg",
]

#full params
againstFull = [
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
    "attack",
    "base_egg_steps",
    "base_happiness",
    "base_total",
    #"capture_rate",
    "defense",
    #"experience_growth",
    #"height_m",
    #"hp",
    "sp_attack",
    "sp_defense",
    "speed",
    #"weight_kg",
]

X = df[against]
y = df['type1']
print("Normal")

#Run selected test
testNeighboursTrainSize(X, y)
testNEstimators(X, y)
testTrainSize(X, y)
testCrossVal(X, y, "rf/cross_val_size.csv")

X = df[againstFull]
testCrossVal(X, y, "rf/cross_val_size_full.csv")
testNeighboursTrainSize(X, y)
testNEstimators(X, y)
testTrainSize(X, y)
testCrossVal(X, y, "rf/cross_val_size.csv")
