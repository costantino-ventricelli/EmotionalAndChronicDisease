# coding=utf-8

import numpy as np
import csv

from os import path
from os import sep
from os import mkdir
from DatasetManager.FileManager import FileManager
from DeepLearningClassifier.MachineLearningModel import MLModel
from DeepLearningClassifier.TaskManager import TaskManager
from DeepLearningClassifier.RHSDistanceExtraction import RHSDistanceExtract

MINIMUM_SAMPLES = 2500
SAMPLES = 50
EXPERIMENT_RESULT = "experiment_result"
LEAVE_ONE_OUT_FOLDER = "leave_one_out"
SAVING_PATHS = EXPERIMENT_RESULT + sep + LEAVE_ONE_OUT_FOLDER


class LeaveOneOutExperiment:

    def __init__(self, dataset):
        self.__dataset = dataset
        dataset = FileManager(self.__dataset)
        self.__patients_paths = dataset.get_patient_paths()
        self.__patients = FileManager.get_ids_from_paths(self.__patients_paths)
        self.__patients.sort()
        self.__ml_model = None
        self.__accuracy_avg = 0
        self.__precision_avg = 0
        self.__recall_avg = 0
        self.__f1_score_avg = 0
        if not path.exists(SAVING_PATHS):
            mkdir(SAVING_PATHS)

    def start_experiment(self, validation_number, test_number):
        print("Leave one out experiment start.")
        for i in range(len(self.__patients)):
            print(i, " iteration:")
            training_paths, validation_paths, test_paths = self.__split_patients(test_number, validation_number)
            training_paths, validation_paths, test_paths = LeaveOneOutExperiment.__create_ml_dataset([training_paths, validation_paths, test_paths])
            feature_extractor = RHSDistanceExtract(MINIMUM_SAMPLES, SAMPLES)
            tensor_training, training_states, training_samples, training_file_samples = feature_extractor.extract_rhs_known_state(training_paths)
            tensor_validation, validation_states, validation_samples, validation_file_samples = feature_extractor.extract_rhs_known_state(validation_paths)
            tensor_test, test_states, test_samples, test_file_samples = feature_extractor.extract_rhs_known_state(test_paths)
            print("Create learning model")
            self.__ml_model = MLModel(tensor_training, training_states, tensor_validation, validation_states, False)
            print("Testing model...")
            # Testo il modello e valuto i risultati
            states_predicted, predicted_results = self.__ml_model.test_model(tensor_test)
            evaluation_result, test_accuracy, test_precision, test_recall, test_f_score, wrong_classified, accuracy_file, \
                    precision_file, recall_file, f1_score_file, wrong_paths = self.__ml_model.classify_results(tensor_test,
                                                                                                               test_states,
                                                                                                               predicted_results,
                                                                                                               states_predicted,
                                                                                                               test_paths,
                                                                                                               test_file_samples)
            self.__accuracy_avg += test_accuracy
            self.__precision_avg += test_precision
            self.__recall_avg += test_recall
            self.__f1_score_avg += test_f_score
            log_file = "iteration_" + str(i) + "_log_file.txt"
            print("Saving result...")
            save_file_path = path.join(SAVING_PATHS, log_file)
            FileManager.log_results(accuracy_file, evaluation_result, f1_score_file, precision_file, recall_file,
                                    save_file_path,
                                    test_accuracy, test_f_score, test_precision, test_recall, wrong_classified,
                                    wrong_paths)

    @staticmethod
    def print_model(ml_model):
        print("Training: ", ml_model[0])
        print("Validation: ", ml_model[1])
        print("Test: ", ml_model[2])

    @staticmethod
    def __create_ml_dataset(ml_model):
        for j in range(len(ml_model)):
            files = []
            for k in range(len(ml_model[j])):
                files = (FileManager.get_files_from_path(ml_model[j][k], files))
            ml_model[j] = FileManager.filter_file(files.copy(), MINIMUM_SAMPLES + 1)
        return ml_model[0], ml_model[1], ml_model[2]

    def __split_patients(self, test_number, validation_number):
        training_paths = self.__patients_paths[0: (len(self.__patients_paths) - validation_number + test_number)]
        validation_paths = self.__patients_paths[len(self.__patients_paths) - (validation_number + test_number):
                                                 len(self.__patients_paths) - test_number]
        test_paths = self.__patients_paths[len(self.__patients_paths) - test_number: len(self.__patients_paths)]
        self.__patients_paths = self.__patients_paths[test_number:] + self.__patients_paths[: test_number]
        return training_paths, validation_paths, test_paths

    def get_metrics(self):
        analysed_file = len(self.__patients_paths)
        return (self.__accuracy_avg / analysed_file), (self.__precision_avg / analysed_file), (self.__recall_avg / analysed_file), \
               (self.__recall_avg / analysed_file)

