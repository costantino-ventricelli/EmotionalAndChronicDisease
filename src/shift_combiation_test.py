# coding=utf-8

from Expreriment import ShiftTaskSelection

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"


def main():
    path_dictionary = {'drawing': 'experiment_result/category_selection/drawing.txt',
                       'matrix': 'experiment_result/category_selection/matrix.txt',
                       'solid_lines': 'experiment_result/category_selection/solid_lines.txt',
                       'trial': 'experiment_result/category_selection/trial.txt',
                       'word_easy': 'experiment_result/category_selection/word_easy.txt',
                       'word_hard': 'experiment_result/category_selection/word_hard.txt'}
    saving_path = 'resource/test_dictionary_file.csv'
    combination = {'matrix': {HEALTHY_STRING: ['_s2.', '_m1.'],
                              DISEASE_STRING: ['_w.', '_m2.']}}
    experiment = ShiftTaskSelection(None, path_dictionary, 2500, 50)
    experiment.start_shift_selection(combination, "matrix_test")


if __name__ == '__main__':
    main()
