# coding=utf-8

import os

from ShallowLearningClassifier import FeatureExtraction
from ShallowLearningClassifier import FeatureSelection


# Questo esperimento serve a selezionare le feature migliori tramite il random forest.


def main():
    feature_selection = FeatureSelection()
    best_hyperparameters, feature = feature_selection.select_feature()
    with open(os.path.join("experiment_result", "experiment_9"), 'w') as file:
        file.write("Number of selected features: " + str(len(feature)) + "\n")
        file.write("Number of non selected features: " + str(len(list(FeatureExtraction.get_file_dictionary()))) + "\n")
        file.write("Selected: " + str(len(feature) / len(list(FeatureExtraction.get_file_dictionary())) * 100) + "% of original feature\n")
        file.write("Best hyperparameters selected for Random Forest Classifier: " + "\n")
        for parameter, value in best_hyperparameters.items():
            file.write("\t" + parameter + ": " + str(value) + "\n")
        file.close()


if __name__ == '__main__':
    main()
