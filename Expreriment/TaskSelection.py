# coding=utf-8

import numpy as np

from DatasetManager import FileManager
from DatasetManager.Costants import *
from DeepLearningClassifier.MachineLearningModel import MLModel
from DeepLearningClassifier.RHSDistanceExtraction import RHSDistanceExtract
from DeepLearningClassifier.TaskManager import TaskManager


class TaskSelection:

    def __init__(self, samples_len, minimum_samples, validation_patients, test_patients):
        self.__tasks = [CLOCK, NATURAL_SENTENCE, PENTAGON, MATRIX_1, MATRIX_2, MATRIX_3, TRIAL_1, T_TRIAL_1, T_TRIAL_2,
                        TRIAL_2, HELLO, V_POINT, H_POINT, SQUARE, SIGNATURE_1, SIGNATURE_2, COPY_SPIRAL, TRACED_SPIRAL,
                        BANK_CHECK, LE, MOM, WINDOW, LISTENING]
        self.__samples_len = samples_len
        self.__minimum_samples = minimum_samples
        self.__validation_patients = validation_patients
        self.__test_patients = test_patients
        self.__best_results = {}
        self.__file_manager = FileManager("Dataset")
        feature_extraction = RHSDistanceExtract(minimum_samples, samples_len)
        for task in self.__tasks:
            paths = TaskManager.get_list_from_task(task, self.__file_manager.get_files_path())
            paths = FileManager.filter_file(paths, min_dim=minimum_samples + 1)
            test_paths = TaskSelection.__get_test_paths(paths)
            validation_number = int(np.ceil(len(paths) * 0.2))
            training_tensor, training_states, _ = feature_extraction.extract_rhs_known_state(paths[0: (len(paths) - validation_number)])
            validation_tensor, validation_states, _ = feature_extraction.extract_rhs_known_state(paths[(len(paths) - validation_number): len(paths)])
            test_tensor, test_states, _ = feature_extraction.extract_rhs_known_state(test_paths)
            machine_learning = MLModel(training_tensor, training_states, validation_tensor, validation_states)
            predicted_results, evaluation, _ = machine_learning.test_model(test_tensor, test_states)
            accuracy, precision, recall, f_score = machine_learning.evaluate_results(predicted_results, test_states)
            if (accuracy, precision, recall, f_score) in self.__best_results:
                tasks = self.__best_results.get((accuracy, precision, recall, f_score), default=None)
                tasks.append(task)
                self.__best_results.update({(accuracy, precision, recall, f_score): tasks})
            else:
                self.__best_results[(accuracy, precision, recall, f_score)] = [task]
        for key, value in self.__best_results.items():
            print("Key: ", key, " Value: ", value)


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

