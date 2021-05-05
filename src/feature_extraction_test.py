# coding=utf-8
import sys
sys.path.append("..")

from ShallowLearningClassifier.FeaturesManager import FeaturesManager

# Questo esperimento avvia la creazione dei file contententi le feature estratte dal dataset:
# resource/feature/...


def main():
    feature = FeaturesManager()
    feature.create_features_file()


if __name__ == '__main__':
    main()
