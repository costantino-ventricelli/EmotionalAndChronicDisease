# coding=utf-8

import csv
import os
from copy import deepcopy

from DatasetManager import HandManager
from ShallowLearningClassifier import ShallowLeaveOneOut

EXPERIMENT_DIRECTORY = os.path.join("experiment_result", "experiment_14")
METRICS_KEY = 0
HEALTHY_INDEX = 1
DISEASE_INDEX = 2
TASK_INDEX = 1
FEATURES = 3

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"

SQUARE_BRACKETS = '[]'

ACCURACY_INDEX = 0
PRECISION_INDEX = 1
RECALL_INDEX = 2
F_SCORE_INDEX = 3


class Experiment14:

    def __init__(self, first_combination, path_directory, saving_file):
        HandManager.set_root_directory()
        healthy_tasks = None
        disease_tasks = None
        self.__saving_path = os.path.join(EXPERIMENT_DIRECTORY, saving_file)
        self.__results = Experiment14.__set_previous_state(saving_file)
        for key, value in first_combination.items():
            self.__task_category = key
            healthy_tasks = value[HEALTHY_STRING]
            disease_tasks = value[DISEASE_STRING]
        if healthy_tasks is not None and not isinstance(healthy_tasks, list):
            healthy_tasks = [healthy_tasks]
        if disease_tasks is not None and not isinstance(disease_tasks, list):
            disease_tasks = [disease_tasks]
        self.__healthy_tasks = healthy_tasks
        self.__disease_tasks = disease_tasks
        self.__base_results = {}
        for _, path in path_directory.items():
            previous_state = Experiment14.__set_previous_state(path)
            key = max(previous_state)
            self.__base_results[key] = previous_state[key]
        if not os.path.exists(EXPERIMENT_DIRECTORY):
            os.mkdir(EXPERIMENT_DIRECTORY)

    def start_selection(self):
        for key, values in self.__base_results.items():
            if key != self.__task_category and key not in self.__results:
                for value in values:
                    if not value[HEALTHY_STRING][0] in self.__healthy_tasks and not value[DISEASE_STRING][0] in self.__disease_tasks:
                        healthy_tasks = deepcopy(self.__healthy_tasks) + value[HEALTHY_STRING]
                        disease_tasks = deepcopy(self.__disease_tasks) + value[DISEASE_STRING]
                        accuracy, precision, recall, f_score = ShallowLeaveOneOut.start_experiment(healthy_tasks, disease_tasks)
                        if os.path.exists(self.__saving_path):
                            file = open(self.__saving_path, 'a')
                        else:
                            file = open(self.__saving_path, 'w')
                        csv_file = csv.writer(file, delimiter=';')
                        csv_file.writerow([(accuracy, precision, recall, f_score), healthy_tasks, disease_tasks])
                        file.close()
            else:
                print("Combination already tried: ", key)

    """
        Questo metodo ha il compito di reimpostare i valori del dizionaro sulla base del file passato al costruttore.
    """
    @staticmethod
    def __set_previous_state(saving_path):
        result = {}
        if os.path.exists(saving_path):
            with open(saving_path, 'r') as file:
                csv_file = csv.reader(file, delimiter=';')
                for row in csv_file:
                    key = eval(row[METRICS_KEY])
                    key = (key[F_SCORE_INDEX], key[RECALL_INDEX], key[PRECISION_INDEX], key[ACCURACY_INDEX])
                    value_dict = {HEALTHY_STRING: Experiment14.__cast_into_list(row[HEALTHY_INDEX]),
                                  DISEASE_STRING: Experiment14.__cast_into_list(row[DISEASE_INDEX])}
                    if key in result.keys():
                        list_of_dict = result.get(key)
                        list_of_dict.append(value_dict)
                    else:
                        result[key] = [value_dict]
                file.close()
        return result

    """
        Questo metodo verifica se una combinazione di task è già stata provata, verificando la sua esistenza nel diziona-
        rio di istanza contenente i risultati della valutazione.
    """
    def __is_already_do(self, healthy_task, disease_task):
        for _, items in self.__results.items():
            for item in items:
                if item[HEALTHY_STRING] == healthy_task and item[DISEASE_STRING] == disease_task:
                    return True
        return False

    """
        Questo metodo mi permette di ottenere una lista di task dalla stringa salvata all'interno del file.
    """
    @staticmethod
    def __cast_into_list(string):
        for character in SQUARE_BRACKETS:
            string = string.replace(character, '')
        strings = string.split(',')
        for i in range(len(strings)):
            strings[i] = strings[i].replace('\'', "").strip()
        return strings
