# coding=utf-8

import os
import csv

from DatasetManager import HandManager
from ShallowLearningClassifier import SVCModel
from ShallowLearningClassifier import FeatureSelection
from ShallowLearningClassifier import CreateDictDataset

EXPERIMENT_DIRECTORY = os.path.join("experiment_result", "experiment_13")
METRICS_KEY = 0
HEALTHY_INDEX = 1
DISEASE_INDEX = 2
TASK_INDEX = 1
FEATURES = 3

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"

SQUARE_BRACKETS = '[]'


class ShallowShiftSelection:

    def __init__(self, first_combination, path_directory=None, saving_path=None):
        HandManager.set_root_directory()
        healthy_tasks = None
        disease_tasks = None
        for key, value in first_combination:
            healthy_tasks = value[HEALTHY_STRING]
            disease_tasks = value[DISEASE_STRING]
        if healthy_tasks is not None and not isinstance(healthy_tasks, list):
            healthy_tasks = [healthy_tasks]
        if disease_tasks is not None and not isinstance(disease_tasks, list):
            disease_tasks = [disease_tasks]
        self.__healthy_tasks = healthy_tasks
        self.__disease_tasks = disease_tasks
        feature_selection = FeatureSelection(healthy_tasks, disease_tasks)
        _, self.__selected_features = feature_selection.select_feature()
        self.__value_task = {}
        if path_directory is not None:
            for key, path in path_directory:
                self.__value_task[key] = max(ShallowShiftSelection.__set_previous_state(path))
        if saving_path is not None:
            self.__results = ShallowShiftSelection.__set_previous_state(saving_path)
        else:
            self.__results = {}
        if not os.path.exists(EXPERIMENT_DIRECTORY):
            os.mkdir(EXPERIMENT_DIRECTORY)

    def start_selection(self):
        print("Start selection")

    """
        Questo metodo ha il compito di reimpostare i valori del dizionaro sulla base del file passato al costruttore.
    """
    @staticmethod
    def __set_previous_state(saving_path):
        with open(saving_path, 'r') as file:
            csv_file = csv.reader(file, delimiter=';')
            result = {}
            for row in csv_file:
                key = eval(row[METRICS_KEY])
                value_dict = {HEALTHY_STRING: ShallowShiftSelection.__cast_into_list(row[HEALTHY_INDEX]),
                              DISEASE_STRING: ShallowShiftSelection.__cast_into_list(row[DISEASE_INDEX])}
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
