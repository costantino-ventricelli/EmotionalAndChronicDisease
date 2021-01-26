# coding=utf-8

from DatasetManager.Costants import *
from Expreriment.EmothawExperiment import EmothawExperiment


def main():
    healthy_tasks = [MOM, LE, SIGNATURE_1, CLOCK]
    disease_tasks = [HELLO, SIGNATURE_2, BANK_CHECK, V_POINT]
    test_tasks = [NATURAL_SENTENCE, PENTAGON]
    emothaw = EmothawExperiment(healthy_tasks, disease_tasks, test_tasks, file_samples=2500)
    emothaw.execute_emothaw_experiment()


if __name__ == "__main__":
    main()
