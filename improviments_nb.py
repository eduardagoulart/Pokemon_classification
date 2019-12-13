import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans


df = pd.read_csv("pokemon.csv")

type_frequency = df.type1.value_counts()


df_class = {
    "water": type_frequency[0],
    "normal": type_frequency[1],
    "grass": type_frequency[2],
    "bug": type_frequency[3],
    "psychic": type_frequency[4],
    "fire": type_frequency[5],
    "rock": type_frequency[6],
    "electric": type_frequency[7],
    "ground": type_frequency[8],
    "poison": type_frequency[9],
    "dark": type_frequency[10],
    "fighting": type_frequency[11],
    "dragon": type_frequency[12],
    "ghost": type_frequency[13],
    "steel": type_frequency[14],
    "ice": type_frequency[15],
    "fairy": type_frequency[16],
    "flying": type_frequency[17],
}

classes = [key for key, value in df_class.items() if value >= 25]
df = df[df['type1'].isin(classes)]

df_X = df[
    [
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
]

df_Y = df[["type1"]]

X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_Y, test_size=0.3, random_state=1
)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print(classification_report(y_test, y_pred))
