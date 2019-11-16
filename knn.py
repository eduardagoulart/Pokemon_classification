import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier



df = pd.read_csv("pokemon.csv")

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

X = df[against]
y = df['type1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

clf = KNeighborsClassifier(n_neighbors=50)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
