# coding=utf-8

import numpy as np

from DatasetManager import FileManager
from DatasetManager.Costants import *
from DeepLearningClassifier import *
from copy import deepcopy


KEY_TUPLE = 0
VALUE_TUPLE = 1


class TaskSelection:

    def __init__(self, samples_len, minimum_samples):
        self.__tasks = [CLOCK, NATURAL_SENTENCE, PENTAGON, MATRIX_1, MATRIX_2, MATRIX_3, TRIAL_1, T_TRIAL_1, T_TRIAL_2,
                        TRIAL_2, HELLO, V_POINT, H_POINT, SQUARE, SIGNATURE_1, SIGNATURE_2, COPY_SPIRAL, TRACED_SPIRAL,
                        BANK_CHECK, LE, MOM, WINDOW, LISTENING]
        self.__samples_len = samples_len
        self.__minimum_samples = minimum_samples
        self.__best_results = {}
        self.__file_manager = FileManager("Dataset")
        self.__feature_extraction = RHSDistanceExtract(minimum_samples, samples_len)
        for task in self.__tasks:
            paths, test_paths, validation_number = self.__select_paths(task)
            test_states, test_tensor, training_states, training_tensor, validation_states, validation_tensor = self.__extract_rhs_segment(
                paths, test_paths, validation_number)
            self.__best_results = TaskSelection.__create_and_evaluate_model(task, test_states, test_tensor, training_states, training_tensor,
                                                                            validation_states, validation_tensor, self.__best_results)

    def execute_task_selection(self):
        previous_max = max(self.__best_results.items())
        actual_tuple = max(self.__best_results.items())
        while previous_max[KEY_TUPLE] <= actual_tuple[KEY_TUPLE]:
            actual_tasks = actual_tuple[VALUE_TUPLE]
            best_results = {}
            for task in self.__tasks:
                actual_tasks.append(task)
                paths, test_paths, validation_number = self.__select_paths(actual_tasks)
                test_states, test_tensor, training_states, training_tensor, validation_states, validation_tensor = self.__extract_rhs_segment(
                    paths, test_paths, validation_number)
                best_results = TaskSelection.__create_and_evaluate_model(actual_tasks, test_states, test_tensor, training_states, training_tensor,
                                                                         validation_states, validation_tensor, best_results)
            previous_max = deepcopy(actual_tuple)
            actual_tuple = max(best_results.items())
        return previous_max

    def __select_paths(self, task):
        paths = TaskManager.get_list_from_task(task, self.__file_manager.get_files_path())
        paths = FileManager.filter_file(paths, min_dim=self.__minimum_samples + 1)
        test_paths = TaskSelection.__get_test_paths(paths)
        validation_number = int(np.ceil(len(paths) * 0.2))
        return paths, test_paths, validation_number

    @staticmethod
    def __get_test_paths(paths):
        test_paths = [paths[0]]
        prev_states = FileManager.get_state_from_id(FileManager.get_id_from_path(paths[0]))
        paths.remove(paths[0])
        i = 0
        while len(test_paths) <= 1:
            if prev_states != FileManager.get_state_from_id(FileManager.get_id_from_path(paths[i])):
                test_paths.append(paths[i])
                paths.remove(paths[i])
            i += 1
        return test_paths

    def __extract_rhs_segment(self, paths, test_paths, validation_number):
        training_tensor, training_states, _ = self.__feature_extraction.extract_rhs_known_state(
            paths[0: (len(paths) - validation_number)])
        validation_tensor, validation_states, _ = self.__feature_extraction.extract_rhs_known_state(
            paths[(len(paths) - validation_number): len(paths)])
        test_tensor, test_states, _ = self.__feature_extraction.extract_rhs_known_state(test_paths)
        return test_states, test_tensor, training_states, training_tensor, validation_states, validation_tensor

    @staticmethod
    def __create_and_evaluate_model(task, test_states, test_tensor, training_states, training_tensor,
                                    validation_states, validation_tensor, best_results):
        machine_learning = MLModel(training_tensor, training_states, validation_tensor, validation_states)
        predicted_results, evaluation, _ = machine_learning.test_model(test_tensor, test_states)
        accuracy, precision, recall, f_score = machine_learning.evaluate_results(predicted_results, test_states)
        best_results = TaskSelection.__fill_dictionary(best_results, accuracy, f_score, precision, recall, task)
        return best_results

    @staticmethod
    def __fill_dictionary(best_results, accuracy, f_score, precision, recall, task):
        if (accuracy, precision, recall, f_score) in best_results:
            tasks = best_results.get((accuracy, precision, recall, f_score), None)
            tasks.append(task)
            best_results.update({(accuracy, precision, recall, f_score): tasks})
        else:
            best_results[(accuracy, precision, recall, f_score)] = [task]
        return best_results

