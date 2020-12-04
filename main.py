from Expreriment.ExperimentModel import Experiment
from DeepLearningClassifier.Costants import *


def main():
    print("Experiment test")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, NATURAL_SENTENCE]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [LISTENING, V_POINT]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_test.txt")
    print("Experiment one")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, PENTAGON]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [NATURAL_SENTENCE, CLOCK]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_1.txt")
    print("Experiment tow")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, V_POINT]
    disease_tasks = [HELLO, SIGNATURE_1, LE, PENTAGON]
    test_tasks = [NATURAL_SENTENCE, CLOCK]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_2.txt")
    print("Experiment three")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, CLOCK]
    disease_tasks = [HELLO, SIGNATURE_1, LE, H_POINT]
    test_tasks = [NATURAL_SENTENCE, PENTAGON]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_3.txt")
    print("Experiment one")
    healthy_tasks = [MOM, LE, SIGNATURE_1, CLOCK]
    disease_tasks = [HELLO, SIGNATURE_2, BANK_CHECK, V_POINT]
    test_tasks = [NATURAL_SENTENCE, PENTAGON]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_4.txt")


if __name__ == "__main__":
    main()
