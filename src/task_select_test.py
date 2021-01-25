# coding=utf-8

from Expreriment import TaskSelection

SAMPLES_LEN = 50
MINIMUM_SAMPLES = 2500
VALIDATION_PATIENTS = 20
TEST_PATIENTS = 1


def main():
    task_selection = TaskSelection(SAMPLES_LEN, MINIMUM_SAMPLES, VALIDATION_PATIENTS, TEST_PATIENTS)


if __name__ == '__main__':
    main()
