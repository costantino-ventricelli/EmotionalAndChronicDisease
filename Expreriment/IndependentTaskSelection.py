# coding=utf-8

from collections import Counter
from copy import deepcopy

import numpy as np
import os
import csv

from DatasetManager import FileManager
from DatasetManager.Costants import *
from DeepLearningClassifier import *

KEY_TUPLE = 0
VALUE_TUPLE = 1
FEATURES = 3


class IndependentTaskSelection:

    def __init__(self, samples_len, minimum_samples):
        self.__tasks = TASKS
        self.__samples_len = samples_len
        self.__minimum_samples = minimum_samples
        self.__best_results = {}
        self.__file_manager = FileManager("Dataset")
        self.__feature_extraction = RHSDistanceExtract(minimum_samples, samples_len)
        self.__ids = FileManager.get_all_ids()
        healthy_ids, disease_ids = FileManager.get_healthy_disease_list()
        directory = os.path.join('experiment_result', 'independent_task_selection')
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(os.path.join(directory, 'init_file.csv'), 'w') as file:
            csv_file = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            csv_file.writerow(['METRICS', 'HEALTHY TASK', 'DISEASE TASK'])
            file.close()
        for healthy_task in self.__tasks:
            healthy_paths = FileManager.get_all_files_ids_tasks(healthy_ids, healthy_task,
                                                                self.__file_manager.get_files_path())
            healthy_paths = FileManager.filter_file(healthy_paths, self.__minimum_samples + 1)
            healthy_validation = int(np.ceil(len(healthy_paths) * 0.20))
            for disease_task in self.__tasks:
                if healthy_task != disease_task:
                    disease_paths = FileManager.get_all_files_ids_tasks(disease_ids, disease_task,
                                                                        self.__file_manager.get_files_path())
                    disease_paths = FileManager.filter_file(disease_paths, self.__minimum_samples + 1)
                    predicted_status = np.zeros(0)
                    theoretical_status = np.zeros(0)
                    for test_id in self.__ids:
                        healthy_paths = IndependentTaskSelection.__delete_files(test_id, healthy_paths)
                        disease_paths = IndependentTaskSelection.__delete_files(test_id, disease_paths)
                        test_tasks = IndependentTaskSelection.__get_tasks_difference(self.__tasks, healthy_task,
                                                                                     disease_task)
                        test_paths = FileManager.get_all_files_ids_tasks([test_id], list(test_tasks),
                                                                         self.__file_manager.get_files_path())
                        test_paths = FileManager.filter_file(test_paths, self.__minimum_samples + 1)
                        disease_validation = int(np.ceil(len(disease_paths) * 0.20))
                        if disease_validation < healthy_validation:
                            validation = disease_validation
                        else:
                            validation = healthy_validation
                        training_tensor, training_states, validation_tensor, validation_states, test_tensor, test_states = \
                            self.__extract_rhs_segment(validation, healthy_paths, disease_paths, test_paths)
                        ml_model = MLModel(training_tensor, training_states, validation_tensor, validation_states)
                        predicted_status_partial, _, _ = ml_model.test_model(test_tensor, test_states)
                        predicted_status = np.concatenate((predicted_status, predicted_status_partial))
                        theoretical_status = np.concatenate((theoretical_status, test_states))
                    accuracy, precision, recall, f_score = MLModel.evaluate_results(predicted_status, theoretical_status)
                    with open(os.path.join(directory, 'init_file.csv'), 'a') as file:
                        csv_file = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                        csv_file.writerow([(accuracy, precision, recall, f_score), healthy_task, disease_task])
                        file.close()

    @staticmethod
    def __get_tasks_difference(tasks, healthy_tasks, disease_tasks):
        if not isinstance(healthy_tasks, list):
            healthy_tasks = [healthy_tasks]
        if not isinstance(disease_tasks, list):
            disease_tasks = [disease_tasks]
        return list(set(tasks).symmetric_difference(healthy_tasks + disease_tasks))

    @staticmethod
    def __delete_files(id, file_list):
        to_delete = FileManager.get_all_file_of_id(id, file_list)
        for file in to_delete:
            del file_list[file_list.index(file)]
        return file_list

    def __extract_rhs_segment(self, validation, healthy_paths, disease_paths, test_paths):
        training_paths = healthy_paths[0: validation] + disease_paths[0: validation]
        validation_paths = healthy_paths[validation: len(healthy_paths)] + disease_paths[validation: len(disease_paths)]
        training_tensor, training_states, _ = self.__feature_extraction.extract_rhs_known_state(training_paths)
        validation_tensor, validation_states, _ = self.__feature_extraction.extract_rhs_known_state(validation_paths)
        test_tensor, test_states, _ = self.__feature_extraction.extract_rhs_known_state(test_paths)
        return training_tensor, training_states, validation_tensor, validation_states, test_tensor, test_states
