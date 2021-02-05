# coding=utf-8
import sys
sys.path.append("..")

from Expreriment import CategoryTaskExtraction
from DatasetManager.Costants import *

SAMPLES_LEN = 50
MINIMUM_SAMPLES = 1000


def main():
    executor = CategoryTaskExtraction(None, [HELLO, SIGNATURE_1, SIGNATURE_2, LE, MOM, WINDOW],
                                      MINIMUM_SAMPLES, SAMPLES_LEN, "simple_word")
    executor.start_selection()


if __name__ == '__main__':
    main()
