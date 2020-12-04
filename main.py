from Expreriment.ExperimentModel import Experiment
from DeepLearningClassifier.Costants import *


def main():
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, PENTAGON]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [NATURAL_SENTENCE, CLOCK]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks)


if __name__ == "__main__":
    main()
