import pandas as pd

df = pd.read_csv('pokemon.csv')
df = df.drop(
    ['japanese_name', 'percentage_male', 'pokedex_number', 'generation', 'height_m', 'base_egg_steps', 'weight_kg'],
    axis=1)
df.to_csv("filtering.csv")
print(df)
print(df.describe())
