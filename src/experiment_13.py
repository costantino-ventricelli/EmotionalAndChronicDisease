# coding=utf-8

import sys
sys.path.append("..")


from Expreriment import Experiment13
from DatasetManager.Costants import *


def main():
    who = sys.argv[1]
    print("Call all selection in one")
    # simple_word
    if who == "simple_word":
        selection("experiment_result/experiment_13/simple_word.csv", [HELLO, SIGNATURE_1, SIGNATURE_2, LE, MOM, WINDOW], "simple_word")
    # drawing
    if who == "drawing":
        selection("experiment_result/experiment_13/drawing.csv", [PENTAGON, CLOCK, SQUARE], "drawing")
    # hard_word
    if who == "hard_word":
        selection("experiment_result/experiment_13/hard_word.csv", [BANK_CHECK, NATURAL_SENTENCE, LISTENING], "hard_word")
    # matrix
    if who == "matrix":
        selection("experiment_result/experiment_13/matrix.csv", [MATRIX_1, MATRIX_2, MATRIX_3], "matrix")
    # trial
    if who == "trial":
        selection("experiment_result/experiment_13/trial.csv", [T_TRIAL_1, TRIAL_1, T_TRIAL_2, TRIAL_2], "trial")
    # solid_lines
    if who == "solid_lines":
        selection("experiment_result/experiment_13/solid_lines.csv", [COPY_SPIRAL, TRACED_SPIRAL, V_POINT, H_POINT], "solid_lines")


def selection(saving_path, task, category):
    extractor = Experiment13(saving_path, task, category)
    extractor.start_healthy_selection()


if __name__ == '__main__':
    main()
