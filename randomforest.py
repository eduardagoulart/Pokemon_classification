import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


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
]

X = df[against]
y = df['type1']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
feature_imp = pd.Series(clf.feature_importances_, index=against).sort_values(ascending=False)
print(feature_imp)