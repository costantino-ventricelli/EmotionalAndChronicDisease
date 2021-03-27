# coding=utf-8

import csv
import os

from DatasetManager import HandManager
from ShallowLearningClassifier import ShallowLeaveOneOut

METRICS_KEY = 0
HEALTHY_VALUE = 1
DISEASE_VALUE = 2

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"
EXPERIMENT_DIRECTORY = os.path.join("experiment_result", "experiment_13")


class Experiment13:

    def __init__(self, prev_file, tasks, category):
        HandManager.set_root_directory()
        if not os.path.exists(EXPERIMENT_DIRECTORY):
            os.mkdir(EXPERIMENT_DIRECTORY)
        self.__tasks = tasks
        self.__prev_file = prev_file
        if prev_file is not None:
            print("Setting previous result")
            self.__results = self.__set_previous_state()
        else:
            self.__results = {}
        self.__category = category

    def start_healthy_selection(self):
        for healthy_task in self.__tasks:
            self.__start_disease_selection(healthy_task)

    def __start_disease_selection(self, healthy_task):
        for disease_task in self.__tasks:
            if healthy_task != disease_task:
                result = self.__start_selection(healthy_task, disease_task)
                if result is None:
                    print("Combination already tried")
                else:
                    print("Selection result:", result)

    def __start_selection(self, healthy_task, disease_task):
        print("Selecting healthy task:", healthy_task, "disease task:", disease_task)
        if not self.__is_already_do(healthy_task, disease_task):
            if not isinstance(healthy_task, list):
                healthy_task = [healthy_task]
            if not isinstance(disease_task, list):
                disease_task = [disease_task]
            accuracy, precision, recall, f_score = ShallowLeaveOneOut.start_experiment(healthy_task, disease_task)
            if self.__prev_file is None:
                self.__prev_file = os.path.join(EXPERIMENT_DIRECTORY, self.__category + ".csv")
                file = open(self.__prev_file, 'w')
            else:
                file = open(self.__prev_file, 'a')
            csv_file = csv.writer(file, delimiter=';')
            csv_file.writerow([(accuracy, precision, recall, f_score), healthy_task, disease_task])
            file.close()
            return {(accuracy, precision, recall, f_score): [healthy_task, disease_task]}
        return None

    """
        Questo metodo verifica se una combinazione di task è già stata provata, verificando la sua esistenza nel diziona-
        rio di istanza contenente i risultati della valutazione.
    """
    def __is_already_do(self, healthy_task, disease_task):
        for _, items in self.__results.items():
            for item in items:
                if item[HEALTHY_STRING].strip() == healthy_task.strip() and item[DISEASE_STRING].strip() == disease_task.strip():
                    return True
        return False

    """
        Questo metodo ha il compito di reimpostare i valori del dizionaro sulla base del file passato al costruttore.
    """
    def __set_previous_state(self):
        with open(self.__prev_file, 'r') as file:
            csv_file = csv.reader(file, delimiter=';')
            result = {}
            for row in csv_file:
                key = eval(row[METRICS_KEY])
                value_dict = {HEALTHY_STRING: row[HEALTHY_VALUE], DISEASE_STRING: row[DISEASE_VALUE]}
                if key in result.keys():
                    list_of_dict = result.get(key)
                    list_of_dict.append(value_dict)
                else:
                    result[key] = [value_dict]
            file.close()
        return result
