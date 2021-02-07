# coding=utf-8

from Expreriment import SelectBestCombinationExperiment


def main():
    path_dictionary = {'drawing': 'experiment_result/category_selection/drawing.txt',
                       'matrix': 'experiment_result/category_selection/matrix.txt',
                       'solid_lines': 'experiment_result/category_selection/solid_lines.txt',
                       'trial': 'experiment_result/category_selection/trial.txt',
                       'word_easy': 'experiment_result/category_selection/word_easy.txt',
                       'word_hard': 'experiment_result/category_selection/word_hard.txt'}
    saving_path = 'experiment_result/linear_selection.txt'
    experiment = SelectBestCombinationExperiment(saving_path, path_dictionary, 2500, 50)
    experiment.start_linear_selection()


if __name__ == '__main__':
    main()
