# coding=utf-8
import sys
sys.path.append("..")

from Expreriment import Experiment3
from DatasetManager.Costants import *

SAMPLES_LEN = 50
MINIMUM_SAMPLES = 1000


def main():
    # L'eseuzione di questo esperimento prevede dierse configurazioni:
    # - simple_word: HELLO, SIGNATURE_1, SIGNATURE_2, LE, MOM, WINDOW;
    # - drawing: PENTAGON, CLOCK, SQUARE;
    # - hard_word: BANK_CHECK, NATURAL_SENTENCE, LISTENING;
    # - matrix: MATRIX_1, MATRIX_2, MATRIX_3;
    # - trial: T_TRIAL_1, TRIAL_1, T_TRIAL_2, TRIAL_2;
    # - solid_lines : COPY_SPIRAL, TRACED_SPIRAL, V_POINT, H_POINT.
    # Rieseguendo lo stesso codice con le diverse configurazioni si otterranno dei file riportanti tutte le combinazioni
    # possibili per i task inviati con i relativi risultati in forma di metriche (accuracy, precision, recall, fscore).
    # I risultati sono disponibili nel percorso experiment_result/experiment_2.
    executor = Experiment3(None, [HELLO, SIGNATURE_1, SIGNATURE_2, LE, MOM, WINDOW],
                           MINIMUM_SAMPLES, SAMPLES_LEN, "simple_word")
    executor.start_healthy_selection()


if __name__ == '__main__':
    main()
