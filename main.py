from Expreriment.ExperimentModel import Experiment
from DeepLearningClassifier.Costants import *


def main():
    print("Experiment test")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, NATURAL_SENTENCE]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [LISTENING, V_POINT]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_test.txt")
    print("Experiment one")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_1, PENTAGON]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [NATURAL_SENTENCE, CLOCK]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_1.txt")


if __name__ == "__main__":
    main()
