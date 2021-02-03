# coding=utf-8

import numpy as np
import os
import csv

from DatasetManager import FileManager
from DatasetManager.Costants import *
from DeepLearningClassifier import *

METRICS_KEY = 0
HEALTHY_VALUE = 1
DISEASE_VALUE = 2
FEATURE = 3

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"


class CategoryTaskExtraction:

    """
        @:param prev_file: contiene il percorso del file frutto di un esecuzione precedente, nel momento in cui viene
                passato None il processo assume che è la prima volta che viene avviata l'analisi
    """
    def __init__(self, prev_file, tasks, minimum_samples, samples_len, category):
        FileManager.set_root_directory()
        self.__tasks = tasks
        # Il file contente i valori calcolati ci permette di ricominciare l'esecuzione del programma nel caso dovesse
        # interrompersi per qualsiasi ragione.
        if prev_file is not None:
            self.__prev_file = prev_file
            self.__results = self.__set_previous_state()
        else:
            self.__prev_file = prev_file
            self.__results = {}
        self.__minimum_samples = minimum_samples
        self.__samples_len = samples_len
        self.__feature_extractor = RHSDistanceExtract(self.__minimum_samples, self.__samples_len)
        self.__category = category

    def star_selection(self):
        print("Starting selection on tasks: ", self.__tasks)
        for healthy_task in self.__tasks:
            self.__start_healthy_tasks(healthy_task)

    def __start_healthy_tasks(self, healthy_task):
        for disease_task in self.__tasks:
            result = self.__start_disease_tasks(healthy_task, disease_task)
            if result is not None:
                print("Execution ended: ", result.items())
            else:
                print("Combination already tried")

    def __start_disease_tasks(self, healthy_task, disease_task):
        if healthy_task != disease_task:
            if not self.__is_already_do(healthy_task, disease_task):
                accuracy, precision, recall, f_score = self.__leave_one_out(healthy_task, disease_task)
                print("Model evaluated")
                directory = os.path.join("experiment_result", "category_selection")
                if not os.path.exists(directory):
                    os.mkdir(directory)
                if self.__prev_file is None:
                    file = open(os.path.join(directory, self.__category + ".txt"), 'w')
                else:
                    file = open(self.__prev_file, 'a')
                csv_file = csv.writer(file, delimiter=';')
                csv_file.writerow([(accuracy, precision, recall, f_score), healthy_task, disease_task])
                file.close()
                return {(accuracy, precision, recall, f_score): [healthy_task, disease_task]}
        return None

    def __leave_one_out(self, healthy_task, disease_task):
        healthy_ids, disease_ids = FileManager.get_healthy_disease_list()
        test_ids = FileManager.get_all_ids()
        file_manager = FileManager("Dataset")
        healthy_file = FileManager.get_all_files_ids_tasks(healthy_ids, healthy_task, file_manager.get_files_path())
        disease_file = FileManager.get_all_files_ids_tasks(disease_ids, disease_task, file_manager.get_files_path())
        healthy_file = FileManager.filter_file(healthy_file, self.__minimum_samples + 1)
        disease_file = FileManager.filter_file(disease_file, self.__minimum_samples + 1)
        validation_number = FileManager.get_validation_number(len(healthy_file), len(disease_file))
        test_tasks = TaskManager.get_tasks_difference(TASKS, healthy_task, disease_task)
        predicted_states = np.zeros(0)
        theoretical_states = np.zeros(0)
        for test_id in test_ids:
            healthy_file_deleted = FileManager.delete_files(test_id, healthy_file)
            disease_file_deleted = FileManager.delete_files(test_id, disease_file)
            training_tensor, training_states, _ = self.__feature_extractor.extract_rhs_known_state(
                healthy_file_deleted[0: len(healthy_file_deleted) - validation_number] + disease_file_deleted[0: len(disease_file_deleted)])
            validation_tensor, validation_states, _ = self.__feature_extractor.extract_rhs_known_state(
                healthy_file_deleted[len(healthy_file_deleted) - validation_number: len(healthy_file_deleted)]
                + disease_file_deleted[len(disease_file_deleted) - validation_number: len(disease_file_deleted)])
            test_file = FileManager.get_all_files_ids_tasks(test_id, test_tasks, file_manager.get_files_path())
            test_file = FileManager.filter_file(test_file, self.__minimum_samples + 1)
            test_tensor = np.zeros((0, self.__samples_len * 2, FEATURE))
            test_states = np.zeros(0)
            for file in test_file:
                partial_tensor, partial_states = self.__feature_extractor.extract_rhs_file(file)
                test_tensor = np.concatenate((test_tensor, partial_tensor))
                test_states = np.concatenate((test_states, partial_states))
            try:
                ml_model = MLModel(training_tensor, training_states, validation_tensor, validation_states)
                predicted_states_partial, evaluation_result, _ = ml_model.test_model(test_tensor, test_states)
                theoretical_states = np.concatenate((theoretical_states, test_states))
                predicted_states = np.concatenate((predicted_states, predicted_states_partial))
            except ValueError as error:
                print("Error: ", error)
                print("Training tensor: ", np.shape(training_tensor), " training states: ", np.shape(training_states))
                print("Validation tensor: ", np.shape(validation_tensor), " validation states: ", np.shape(validation_states))
                print("Test tensor: ", np.shape(test_tensor), " test states; ", np.shape(test_states))
                print("Validation number: ", validation_number)
                print("Healthy file: ", len(healthy_file_deleted), " Disease file: ", len(disease_file_deleted),
                      " Test id: ", test_id)
        return MLModel.evaluate_results(predicted_states, theoretical_states)

    def __is_already_do(self, healthy_task, disease_task):
        for _, items in self.__results.items():
            for item in items:
                if item[HEALTHY_STRING].strip() == healthy_task.strip() and item[DISEASE_STRING].strip() == disease_task.strip():
                    return True
        return False

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
