# coding=utf-8

import sys
import os
import csv

sys.path.append("..")

from Expreriment import ShallowShiftSelection
from DatasetManager import HandManager
from DatasetManager.Costants import *
from copy import deepcopy

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"

# L'esperimento 13 permette di selezionare per categoria i task


def main():
    combination = {'matrix': {HEALTHY_STRING: ['_s2.', '_m1.'],
                              DISEASE_STRING: ['_w.', '_m2.']}}
    path_dictionary = {'drawing': 'experiment_result/experiment_2/drawing.txt',
                       'matrix': 'experiment_result/experiment_2/matrix.txt',
                       'solid_lines': 'experiment_result/experiment_2/solid_lines.txt',
                       'trial': 'experiment_result/experiment_2/trial.txt',
                       'word_easy': 'experiment_result/experiment_2/word_easy.txt',
                       'word_hard': 'experiment_result/experiment_2/word_hard.txt'}
    shallow_selection = ShallowShiftSelection(first_combination=combination, )


if __name__ == '__main__':
    main()