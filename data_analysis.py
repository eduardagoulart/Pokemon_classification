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
df.to_csv('filtering.csv')
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
    plt.hist(x, bins=bins_vec)
    plt.show()


