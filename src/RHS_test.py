# coding=utf-8

import sys
sys.path.append("..")

from DatasetManager.Costants import *
from Expreriment.RHSExperimentModel import Experiment


def main():
    print("Start experiment suite")
    experiment_test()
    """experiment_one()
    experiment_tow()
    experiment_three()
    experiment_four()
    experiment_five()
    experiment_six()
    experiment_seven()
    experiment_eight()"""


def experiment_test():
    print("Experiment test")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, NATURAL_SENTENCE]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [LISTENING, V_POINT]
    experiment = Experiment()
    experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, 2500, "experiment_multilayer.txt")


def experiment_one():
    print("Experiment one")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, PENTAGON]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [NATURAL_SENTENCE, CLOCK]
    experiment = Experiment()
    experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, 2500, "experiment_1_multilayer.txt")


def experiment_tow():
    print("Experiment tow")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, V_POINT]
    disease_tasks = [HELLO, SIGNATURE_1, LE, PENTAGON]
    test_tasks = [NATURAL_SENTENCE, CLOCK]
    experiment = Experiment()
    experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, 2500, "experiment_2_multilayer.txt")


def experiment_three():
    print("Experiment three")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, CLOCK]
    disease_tasks = [HELLO, SIGNATURE_1, LE, H_POINT]
    test_tasks = [NATURAL_SENTENCE, PENTAGON]
    experiment = Experiment()
    experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, 2500, "experiment_3_multilayer.txt")


def experiment_four():
    print("Experiment four")
    healthy_tasks = [MOM, LE, SIGNATURE_1, CLOCK]
    disease_tasks = [HELLO, SIGNATURE_2, BANK_CHECK, V_POINT]
    test_tasks = [NATURAL_SENTENCE, PENTAGON]
    experiment = Experiment()
    experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, 2500, "experiment_4_multilayer.txt")


def experiment_five():
    print("Experiment five")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, NATURAL_SENTENCE]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [LISTENING, CLOCK]
    experiment = Experiment()
    experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, 2500, "experiment_5_multilayer.txt")


def experiment_six():
    print("Experiment six")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, NATURAL_SENTENCE]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [LISTENING]
    experiment = Experiment()
    experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, 2500, "experiment_6_multilayer.txt")


def experiment_seven():
    print("Experiment seven")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, CLOCK]
    disease_tasks = [HELLO, SIGNATURE_1, LE, PENTAGON]
    test_tasks = [CLOCK, PENTAGON]
    experiment = Experiment()
    experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, 2500, "experiment_7_multilayer.txt")


def experiment_eight():
    print("Experiment eight")
    healthy_tasks = [MOM, LE, NATURAL_SENTENCE, CLOCK]
    disease_tasks = [HELLO, LISTENING, BANK_CHECK, V_POINT]
    test_tasks = [NATURAL_SENTENCE, LISTENING]
    experiment = Experiment()
    experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, 2500, "experiment_8_multilayer.txt")


if __name__ == "__main__":
    main()
