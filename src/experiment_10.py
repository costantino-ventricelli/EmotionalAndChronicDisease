# coding=utf-8
import sys
sys.path.append("..")

import os

from Expreriment import ShallowLeaveOneOut


EXPERIMENT_RESULT = os.path.join("experiment_result", "experiment_10.txt")

# In questo esperimento si avvia il modello di test DeepLeaveOneOut per la SVM.


def main():
    accuracy, precision, recall, f_score = ShallowLeaveOneOut.start_experiment()
    with open(EXPERIMENT_RESULT, 'w') as file:
        file.write("Leave One Out Results\n")
        file.write("ACCURACY: " + str(accuracy * 100) + "\n")
        file.write("PRECISION: " + str(precision * 100) + "\n")
        file.write("RECALL: " + str(recall * 100) + "\n")
        file.write("F_SCORE: " + str(f_score * 100) + "\n")
        file.close()


if __name__ == '__main__':
    main()
