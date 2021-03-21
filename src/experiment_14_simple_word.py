# coding=utf-8

import sys

sys.path.append("..")

from Expreriment import ShallowShiftSelection
from playsound import playsound

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"

# L'esperimento 14 permette di selezionare per categoria i task


def main():
    combination = {'word_easy': {HEALTHY_STRING: ['_s2.', '_tmtt1.', '_sc.', '_vp.', '_m2.'],
                             DISEASE_STRING: ['_le.', '_tmt1.', '_cdt.', '_cs.', '_m1.']}}
    path_dictionary = {'drawing': 'experiment_result/experiment_13/drawing.csv',
                       'matrix': 'experiment_result/experiment_13/matrix.csv',
                       'solid_lines': 'experiment_result/experiment_13/solid_lines.csv',
                       'trial': 'experiment_result/experiment_13/trial.csv',
                       'word_easy': 'experiment_result/experiment_13/simple_word.csv',
                       'word_hard': 'experiment_result/experiment_13/hard_word.csv'
                       }
    shallow_selection = ShallowShiftSelection(first_combination=combination, path_directory=path_dictionary,
                                              saving_file='word_easy_5.csv')
    shallow_selection.start_selection()
    playsound('resource/google_glass_success.mp3')


if __name__ == '__main__':
    main()