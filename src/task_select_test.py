# coding=utf-8
import sys
sys.path.append("..")

from Expreriment import TaskSelection

SAMPLES_LEN = 50
MINIMUM_SAMPLES = 2000


def main():
    task_selection = TaskSelection(SAMPLES_LEN, MINIMUM_SAMPLES)
    best_task = task_selection.execute_task_selection()
    print("Best task combination (text dependent): ", best_task)


if __name__ == '__main__':
    main()
