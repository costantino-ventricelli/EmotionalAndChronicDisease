# coding=utf-8
import sys
sys.path.append("..")

from Expreriment import CategoryTaskExtraction
from DatasetManager import Costants

SAMPLES_LEN = 50
MINIMUM_SAMPLES = 2000


def main():
    executor = CategoryTaskExtraction(None, [Costants.MATRIX_1, Costants.MATRIX_2, Costants.MATRIX_3], 2500, 50, "matrix")
    executor.star_selection()


if __name__ == '__main__':
    main()
