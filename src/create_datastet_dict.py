# coding=utf-8


from ShallowLearningClassifier import CreateDictDataset
from ShallowLearningClassifier import FeatureSelection


def main():
    fs = FeatureSelection()
    _, features = fs.select_feature()
    CreateDictDataset(features)


if __name__ == '__main__':
    main()
