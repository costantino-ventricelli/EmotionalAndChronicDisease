# coding=utf-8

from SupportVectorMachineClassifier.FeaturesManager import FeaturesManager
from sklearn.neighbors import KernelDensity
import numpy as np


def main():
    samples = np.linspace(0, 1, 1000).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(samples)
    feature = FeaturesManager()
    feature.create_features_file()


if __name__ == '__main__':
    main()
