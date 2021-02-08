# coding=utf-8

import csv
import os

from DeepLearningClassifier import LeaveOneOut
from DatasetManager import FileManager

METRICS_KEY = 0
HEALTHY_INDEX = 1
DISEASE_INDEX = 2
TASK_INDEX = 1
FEATURES = 3

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"

SQUARE_BRACKETS = '[]'


class SelectBestCombinationExperiment:

    """
        @:param saving_path: indica il percorso del file dove sono salvati i risultati della precedente esecuzione.
        @:param path_dictionary: indica i percorsi dove sono contenuti i risultati della selezione di categoria.
        @:param minumum_samples: indica il numero minimo di righe del file.
        @:param samples_len: indica il numero di campioni prelevati per ogni campione
    """
    def __init__(self, saving_path, path_dictionary, minimum_samples, samples_len):
        FileManager.set_root_directory()
        self.__saving_path = saving_path
        self.__minimum_samples = minimum_samples
        self.__samples_len = samples_len
        # Se è stato impostato il percorso dell'esecuzione precedente dovrò riporate il sistema allo stato indicato nel
        # file.
        if self.__saving_path is not None:
            self.__results = self.__set_previous_state(self.__saving_path)
        else:
            self.__results = {}
        self.__value_tasks = {}
        # Qui viene prelevato il massimo di ogni categoria selezionata.
        for key, item in path_dictionary.items():
            self.__value_tasks[key] = max(SelectBestCombinationExperiment.__set_previous_state(item).items())[TASK_INDEX]

    def start_linear_selection(self):
        healthy = []
        disease = []
        learning_method = LeaveOneOut(self.__minimum_samples, self.__samples_len, FEATURES, "Dataset")
        file_path = os.path.join("experiment_result", "linear_selection.txt")
        # Ripeto per tutte le categorie selezionate.
        for _, values in self.__value_tasks.items():
            # Per ogni valore, nel caso ci siano piu combinazioni che danno lo stesso risultato
            for value in values:
                # Questo blocco if-else mi permette di trattare allo stesso modo liste di task o task singoli.
                if not isinstance(value[HEALTHY_STRING], list):
                    healthy.append(value[HEALTHY_STRING])
                    disease.append(value[DISEASE_STRING])
                else:
                    healthy += value[HEALTHY_STRING]
                    disease += value[DISEASE_STRING]
                # Qui verifico che la combinazione selezionata non sia stata già testata.
                if not self.__is_already_do(healthy, disease):
                    accuracy, precision, recall, f_score = learning_method.leave_one_out(healthy, disease)
                    print("METRICS: ", accuracy, ", ", precision, ", ", recall, ", ", f_score)
                    if os.path.exists(file_path):
                        mode = 'a'
                    else:
                        mode = 'w'
                    with open(file_path, mode) as file:
                        csv_file = csv.writer(file, delimiter=';')
                        csv_file.writerow([(accuracy, precision, recall, f_score), healthy, disease])
                        file.close()
                else:
                    print("Selection already do...")

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
                value_dict = {HEALTHY_STRING: SelectBestCombinationExperiment.__cast_into_list(row[HEALTHY_INDEX]),
                              DISEASE_STRING: SelectBestCombinationExperiment.__cast_into_list(row[DISEASE_INDEX])}
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
