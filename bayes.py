import numpy as np
import pandas as pd
import math


def calculate_mean(train_data, features, target):
    classes = {}

    for class_name in train_data[target].unique():
        summary = {}

        for feature in features:
            summary[feature] = [
                train_data[feature][train_data[target] == class_name].mean(),
                train_data[feature][train_data[target] == class_name].std(),
            ]

        classes[class_name] = summary
    return classes


def probability(x, mean, standard_deviation):
    if standard_deviation == 0:
        return 0
    return (1 / (math.sqrt(2 * math.pi) * standard_deviation)) * math.exp(
        -(math.pow(x - mean, 2) / (2 * math.pow(standard_deviation, 2)))
    )


def gauss_classes(classes, test_data, train_data, target, features):
    for idx in test_data.index:
        chances = {}
        for class_name in train_data[target].unique():
            chance_count = 1
            for feature in features:

                chance_count += np.log(
                    probability(
                        test_data.loc[idx, feature],
                        classes[class_name][feature][0],
                        classes[class_name][feature][1],
                    )
                )

            chances[class_name] = chance_count

        test_data.loc[idx, "classify"] = max(chances, key=chances.get)

    accuracy = len(
        test_data[test_data[target] == test_data["classify"]].index
    ) / len(test_data.index)

    return test_data, accuracy


def naive_bayes(test_data, train_data, features, target):

    return gauss_classes(
        calculate_mean(train_data, features, target),
        test_data,
        train_data,
        target,
        features,
    )


def main():
    df = pd.read_csv("./pokemon.csv")
    target = "type1"

    features = [
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

    train_test = np.random.rand(len(df)) < 0.90
    train_data = df[train_test]
    test_data = df[~train_test]

    test_data["classify"] = test_data.index

    test_data_bayes, accuracy_bayes = naive_bayes(
        test_data, train_data, features, target
    )

    print(f"Naive Bayes: {accuracy_bayes*100:.2f}%")


if __name__ == "__main__":
    main()
