import pandas as pd

"""
Naive Bayes calcula todas as probabilidades em realação à classe. Ou seja, o que deverá ser feito:
P(against_bug=good|type1=glass)
"""

df = pd.read_csv("pokemon.csv")
df_len = len(df)
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

print(df_class)

probability_class = {}
for type1, freq in df_class.items():
    probability_class[type1] = freq / df_len


def against(df, column1, column2):
    against = df.groupby([column1, "type1"]).count()
    against.reset_index(level=0, inplace=True)
    against.reset_index(level=0, inplace=True)
    against = against[["type1", column1, column2]]

    return against.rename(columns={column1: "against", column2: "probability"})

def probability(df, type1, against):
    value = df[(df.type1 == type1) & (df.against == against)]
    print(value)
    return value['probability'] / df_class[type1]

# print(against(df, "against_dark", "is_legendary"))
against_df = against(df, "against_dark", "is_legendary")
print(probability(against_df, "ice", 1.0))