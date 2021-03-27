# coding=utf-8

import sys

sys.path.append("..")

from Expreriment import Experiment14
from playsound import playsound

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"

# L'esperimento 14 permette di selezionare per categoria i task


def main():
    combination = {'drawing': {HEALTHY_STRING: ['_sc.', '_vp.', '_tmtt1.', '_s2.', '_ds.'],
                              DISEASE_STRING: ['_cdt.', '_cs.', '_tmt1.', '_le.', '_sw.']}}
    path_dictionary = {'drawing': 'experiment_result/experiment_13/drawing.csv',
                       'matrix': 'experiment_result/experiment_13/matrix.csv',
                       'solid_lines': 'experiment_result/experiment_13/solid_lines.csv',
                       'trial': 'experiment_result/experiment_13/trial.csv',
                       'word_easy': 'experiment_result/experiment_13/simple_word.csv',
                       'word_hard': 'experiment_result/experiment_13/hard_word.csv'
                       }
    shallow_selection = Experiment14(first_combination=combination, path_directory=path_dictionary,
                                     saving_file='drawing_5.csv')
    shallow_selection.start_selection()
    playsound('resource/google_glass_success.mp3')


if __name__ == '__main__':
    main()