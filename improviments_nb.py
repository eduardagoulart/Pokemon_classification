import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans


df = pd.read_csv("pokemon.csv")
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

kmeans = KMeans(n_clusters=8, random_state=0).fit(df_X)

X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_Y, test_size=0.3, random_state=1
)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
