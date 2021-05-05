# coding=utf-8
import sys
sys.path.append("..")

from Expreriment import Experiment8

SAMPLES_LEN = 50
MINIMUM_SAMPLES = 2000


def main():
    task_selection = Experiment8(SAMPLES_LEN, MINIMUM_SAMPLES)
    best_task = task_selection.execute_simple_task_selection()
    print("Best tasks combination (text dependent): ", best_task)


if __name__ == '__main__':
    main()
