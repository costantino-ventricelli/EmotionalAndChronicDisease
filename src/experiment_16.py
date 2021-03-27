# coding=utf-8

import os
import sys

sys.path.append('..')

from Expreriment import Experiment16

EXPERIMENT_RESULT = os.path.join("experiment_result", "experiment_16.txt")

# Questo esperimento avvia la costruzione del modello selezinando tutti i task contemporaneamente.


def main():
    experiment_16 = Experiment16(age_min=75, age_max=100)
    accuracy, precision, recall, f_score = experiment_16.start_validation()
    with open(EXPERIMENT_RESULT, 'w') as file:
        file.write("accuracy: " + str(accuracy) + "\n")
        file.write("precision: " + str(precision) + "\n")
        file.write("recall: " + str(recall) + "\n")
        file.write("f_score: " + str(f_score) + "\n")
        file.close()


if __name__ == '__main__':
    main()
