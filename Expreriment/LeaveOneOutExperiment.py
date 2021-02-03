# coding=utf-8

from collections import Counter
import os

import numpy as np

from DatasetManager.FileManager import FileManager
from DeepLearningClassifier.MachineLearningModel import MLModel
from DeepLearningClassifier.RHSDistanceExtraction import RHSDistanceExtract

MINIMUM_SAMPLES = 2500
SAMPLES = 50
FEATURES = 3
EXPERIMENT_RESULT = "experiment_result"


class LeaveOneOutExperiment:

    def __init__(self, dataset):
        self.__dataset = dataset
        dataset = FileManager(self.__dataset)
        self.__patients_paths = dataset.get_files_path()
        self.__patients_paths = FileManager.filter_file(self.__patients_paths, MINIMUM_SAMPLES)
        self.__patients = FileManager.get_ids_from_dir(dataset.get_patient_paths())
        self.__patients.sort()

    def start_experiment(self):
        print("Leave one out experiment start.")
        feature_extraction = RHSDistanceExtract(MINIMUM_SAMPLES, SAMPLES)
        global_results = np.zeros(0)
        global_states = np.zeros(0)
        for test_id in self.__patients:
            print("Test id: ", test_id)
            print("Deleting test file...")
            deleted_paths = FileManager.delete_files(test_id, self.__patients_paths)
            validation_number = int(np.ceil(len(deleted_paths) * 0.20))
            training_file = deleted_paths[0: len(deleted_paths) - validation_number]
            validation_file = deleted_paths[len(deleted_paths) - validation_number: len(deleted_paths)]
            test_file = FileManager.get_all_file_of_id(test_id, self.__patients_paths)
            print("Creating training tensor...")
            training_tensor, training_states, _ = feature_extraction.extract_rhs_known_state(training_file)
            print("Creating validation tensor...")
            validation_tensor, validation_states, _ = feature_extraction.extract_rhs_known_state(validation_file)
            test_tensor = np.zeros((0, SAMPLES * 2, FEATURES))
            test_states = np.zeros(0)
            print("Creating test tensor...")
            for file in test_file:
                partial_tensor, partial_states = feature_extraction.extract_rhs_file(file)
                partial_states = np.concatenate((test_tensor, partial_tensor))
                test_states = np.concatenate((test_states, partial_states))
            print("Creating model...")
            ml_model = MLModel(training_tensor, training_states, validation_tensor, validation_states)
            print("Testing model...")
            partial_results, _, _ = ml_model.test_model(test_tensor, test_states)
            accuracy, _ = MLModel.get_accuracy_precision(partial_results, test_states)
            print("Accuracy: ", accuracy)
            print("Update results...")
            global_results = np.concatenate((global_results, partial_results))
            global_states = np.concatenate((global_states, test_states))
        with open(os.path.join(EXPERIMENT_RESULT, "leave_one_out.txt", 'w')) as file:
            accuracy, precision, recall, f_score = MLModel.evaluate_results(global_results, global_states)
            file.write("LEAVE ONE OUT EXPERIMENT:\n")
            file.write("ACCURACY: " + str(accuracy * 100) + "\n")
            file.write("PRECISION: " + str(precision * 100) + "\n")
            file.write("RECALL: " + str(recall * 100) + "\n")
            file.write("F_SCORE: " + str(f_score * 100) + "\n")
            file.close()