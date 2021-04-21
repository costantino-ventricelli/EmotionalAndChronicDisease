# coding=utf-8

import csv
import os
import sys

sys.path.append('..')

from Expreriment import Experiment16
from DatasetManager.Costants import *

# pip3 install EMD-signal differint playsound

matrix = [MATRIX_1, MATRIX_2, MATRIX_3]
drawing = [CLOCK, PENTAGON, SQUARE]
solid_lines = [H_POINT, V_POINT, COPY_SPIRAL, TRACED_SPIRAL]
trial = [TRIAL_1, T_TRIAL_1, TRIAL_2, T_TRIAL_2]
word_easy = [HELLO, SIGNATURE_1, SIGNATURE_2, LE, MOM, WINDOW]
word_hard = [NATURAL_SENTENCE, LISTENING, BANK_CHECK]

EXPERIMENT_RESULT = os.path.join("experiment_result", "experiment_17")

# In questo esperimento basterà cambiare il vettore dei task e il file di salvataggio per ottenere tutte le combinazioni
# dei task sui tre stati rendendoli task independent in quanto per il task di test verranno selezionati tutti i task che
# non sono stati utilizzati per l'addestramento.


def main():
    if not os.path.exists(EXPERIMENT_RESULT):
        os.mkdir(EXPERIMENT_RESULT)
    experiment = Experiment16(age_min=75, age_max=100)
    for healthy_task in matrix:
        for mild_task in matrix:
            if mild_task != healthy_task:
                for disease_task in matrix:
                    if disease_task != healthy_task and disease_task != mild_task:
                        accuracy, precision, recall, f_score = experiment.start_validation(healthy_task, mild_task, disease_task)
                        if os.path.exists(os.path.join(EXPERIMENT_RESULT, 'matrix.csv')):
                            file = open(os.path.join(EXPERIMENT_RESULT, 'matrix.csv'), 'a')
                        else:
                            file = open(os.path.join(EXPERIMENT_RESULT, 'matrix.csv'), 'w')
                        csv_file = csv.writer(file, delimiter=';')
                        csv_file.writerow([(accuracy, precision, recall, f_score), healthy_task, mild_task, disease_task])
                        file.close()


if __name__ == '__main__':
    main()