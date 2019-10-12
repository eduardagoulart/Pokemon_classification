import pandas as pd
import operator

inputs = [  # Esse input vai ser dado pelo usuário, apenas um caso teste
    1.0,
    1.0,
    1.0,
    0.5,
    0.5,
    0.5,
    2.0,
    2.0,
    1.0,
    0.25,
    1.0,
    2.0,
    1.0,
    1.0,
    2.0,
    1.0,
    1.0,
    0.5,
]

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

hash_change_value = {
    "against_bug": "bug",
    "against_dark": "dark",
    "against_dragon": "dragon",
    "against_electric": "electric",
    "against_fairy": "fairy",
    "against_fight": "fight",
    "against_fire": "fire",
    "against_flying": "flying",
    "against_ghost": "ghost",
    "against_grass": "grass",
    "against_ground": "ground",
    "against_ice": "ice",
    "against_normal": "normal",
    "against_poison": "poison",
    "against_psychic": "psychic",
    "against_rock": "rock",
    "against_steel": "steel",
    "against_water": "water",
}


def class_probability(_class):
    return df_class[_class] / len(df)


def probability_elem(feature, value, class_value, p_class, m):
    n = len(df[df[feature] == value])
    n_c = len(df[(df.type1 == class_value) & (df[feature] == value)])
    return (n_c + (p_class * m)) / (m + n)
    # print(df[df[feature] == value])


info = hash_change_value[against[0]]
# print(df_class[info])
print(probability_elem(against[0], inputs[0], info, class_probability(info), m=5))
"""
probability = {}
for key, item in df_class.items():
    prob = []
    # item = size of the class
    for i in range(len(inputs)):
        # print('------------' * 5)
        # print(df[against[i]])
        new_df = df[(df.type1 == key) & (df[against[i]] == inputs[i])]
        value1 = len(new_df)
        # print(f'{key} -> {item} = {value1}')
        prob.append(value1 / item)

    aux = probability_class[key]
    for i in prob:
        aux *= i
    probability[key] = aux

print(probability)
sorted_d = dict(sorted(probability.items(), key=operator.itemgetter(0)))
print(sorted_d)
result = list(sorted_d.keys())[0]
print(probability[result])
final_prob = (probability[result] / sum(probability.values())) * 100
print(f"Probabilidade de {final_prob} de ser {result}")

# value1 = len(new_df)
# value2 = df_class["water"]
# print(f"probability = {value1 / value2}")
"""
