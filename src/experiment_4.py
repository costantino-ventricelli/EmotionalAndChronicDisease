# coding=utf-8
import sys
sys.path.append("..")

from Expreriment import SelectTask
from DeepLearningClassifier import RHSDistanceExtract


# Questo esperimento genera e testa il modello selezionando un singolo task per effettuare addestramento, validazione e
# test, senza differenziare il task per gli utenti sani e malati, in modo da individuare quale dei task mostra i migliori
# risultati.


def main():
    select_task = SelectTask('best_task.txt', 2500, 50, RHSDistanceExtract(2500, 50), feature=3)
    select_task.select_task()


if __name__ == '__main__':
    main()
