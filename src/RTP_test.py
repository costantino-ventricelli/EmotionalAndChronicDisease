from DatasetManager.Costants import *
from Expreriment.RTPExperimentModel import Experiment


def main():
    print("Experiment test")
    healthy_tasks = [MOM, BANK_CHECK, SIGNATURE_2, NATURAL_SENTENCE]
    disease_tasks = [HELLO, SIGNATURE_1, LE, WINDOW]
    test_tasks = [LISTENING, V_POINT]
    experiment = Experiment()
    experiment.execute_experiment("Dataset", healthy_tasks, disease_tasks, test_tasks, 2500,
                                  "rtp_experiment.txt")


if __name__ == "__main__":
    main()
