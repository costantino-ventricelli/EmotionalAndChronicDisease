# coding=utf-8

from SupportVectorMachineClassifier import FeatureSelection
from SupportVectorMachineClassifier import FeatureExtraction


# Questo esperimento serve a selezionare le feature migliori tramite il random forest.


def main():
    feature_selection = FeatureSelection()
    feature = feature_selection.select_feature()
    print("Number of selected features: ", len(feature))
    print("Number of non selected features: ", len(list(FeatureExtraction.get_file_dictionary())))


if __name__ == '__main__':
    main()
