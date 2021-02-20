# coding=utf-8
import sys
sys.path.append("..")

from SupportVectorMachineClassifier.FeaturesManager import FeaturesManager


def main():
    feature = FeaturesManager()
    feature.create_features_file()


if __name__ == '__main__':
    main()
