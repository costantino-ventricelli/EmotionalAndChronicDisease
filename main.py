from DeepLearningClassifier.Costants import *
from Expreriment.ExperimentModel import Experiment


def main():
    experiment_test()
    experiment_one()
    experiment_tow()
    experiment_three()
    experiment_four()
    experiment_five()
    experiment_six()


def experiment_test():
    print("Experiment test")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, NATURAL_SENTENCE]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [LISTENING, V_POINT]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_0.txt")


def experiment_one():
    print("Experiment one")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, PENTAGON]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [NATURAL_SENTENCE, CLOCK]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_1.txt")


def experiment_tow():
    print("Experiment tow")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, V_POINT]
    disease_tasks = [HELLO, SIGNATURE_1, LE, PENTAGON]
    test_tasks = [NATURAL_SENTENCE, CLOCK]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_2.txt")


def experiment_three():
    print("Experiment three")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, CLOCK]
    disease_tasks = [HELLO, SIGNATURE_1, LE, H_POINT]
    test_tasks = [NATURAL_SENTENCE, PENTAGON]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_3.txt")


def experiment_four():
    print("Experiment four")
    healthy_tasks = [MOM, LE, SIGNATURE_1, CLOCK]
    disease_tasks = [HELLO, SIGNATURE_2, BANK_CHECK, V_POINT]
    test_tasks = [NATURAL_SENTENCE, PENTAGON]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_4.txt")


def experiment_five():
    print("Experiment five")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, NATURAL_SENTENCE]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [LISTENING, CLOCK]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_5.txt")


def experiment_six():
    print("Experiment six")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, NATURAL_SENTENCE]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [LISTENING]
    Experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, "experiment_6.txt")


if __name__ == "__main__":
    main()
