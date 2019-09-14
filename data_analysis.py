import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('pokemon.csv')
df = df.drop(
    ['japanese_name', 'percentage_male', 'pokedex_number', 'generation', 'height_m', 'base_egg_steps', 'weight_kg',
     'experience_growth', 'speed', 'base_happiness', 'base_total', 'is_legendary', 'sp_defense', 'sp_attack', 'name',
     'hp', "defense", 'abilities', 'attack', 'capture_rate'], axis=1)

print(df)
sns.set(color_codes=True)


def types_corr():
    bug = [df[df.type1 == 'bug']]
    pearson_corr = df.corr(method='pearson')
    spearman_corr = df.corr(method='spearman')
    kendall_corr = df.corr(method='kendall')
    pearson_corr.to_csv('corr.csv')


def against_bug_type1_count():
    """
    How against bug distribution with all type1 class
    :return: graph
    """
    type1 = df['type1'].drop_duplicates().tolist()
    type1_count = df.groupby("type1").count()
    type1_count['type1'] = type1
    type1_count.plot(kind='bar', x='type1', y='against_bug')
    plt.show()


def against_bug_distribution():
    x = np.array(df['against_bug'])
    bins_vec = []
    for i in x:
        if i not in bins_vec:
            bins_vec.append(i)

    bins_vec = sorted(bins_vec)
    print(bins_vec)
    plt.hist(x, bins=bins_vec)
    plt.show()


def against_dark_distribution():
    x = np.array(df['against_dark'])
    bins_vec = []
    for i in x:
        if i not in bins_vec:
            bins_vec.append(i)

    bins_vec = sorted(bins_vec)
    print(bins_vec)
    plt.hist(x, bins=bins_vec)
    plt.show()


print(df)

df["against_bug"], df["against_dark"], df["against_dragon"], df["against_electric"], df["against_fairy"], df[
    "against_fight"], df["against_fire"], df["against_flying"], df["against_ghost"], df["against_grass"], df[
    "against_ground"], df["against_ice"], df["against_normal"], df["against_poison"], df["against_psychic"], df[
    "against_steel"], df["against_rock"], df["against_water"] = df["against_bug"].replace([0.25, 0.5, 1.00, 2.00, 4.00],
                                                                                          ["terrible", "bad", "inter",
                                                                                           "good", "amazing"]), df[
                                                                    "against_dark"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_dragon"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_electric"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_fairy"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_fire"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_flying"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_ghost"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_grass"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_ground"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_ice"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_normal"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_poison"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_psychic"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_steel"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_rock"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_water"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]), df["against_water"].replace(
    [0.25, 0.5, 1.00, 2.00, 4.00], ["terrible", "bad", "inter", "good", "amazing"]),

df.to_csv('filtering.csv')
