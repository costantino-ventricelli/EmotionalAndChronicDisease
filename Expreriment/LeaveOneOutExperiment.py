# coding=utf-8

from collections import Counter
from os import mkdir
from os import path
from os import sep

import numpy as np

from DatasetManager.FileManager import FileManager
from DeepLearningClassifier.MachineLearningModel import MLModel
from DeepLearningClassifier.RHSDistanceExtraction import RHSDistanceExtract

MINIMUM_SAMPLES = 2500
SAMPLES = 50
EXPERIMENT_RESULT = "experiment_result"
LEAVE_ONE_OUT_FOLDER = "leave_one_out"
SAVING_PATHS = sep + EXPERIMENT_RESULT + sep + LEAVE_ONE_OUT_FOLDER


class LeaveOneOutExperiment:

    def __init__(self, dataset):
        self.__dataset = dataset
        dataset = FileManager(self.__dataset)
        self.__patients_paths = dataset.get_patient_paths()
        self.__patients = FileManager.get_ids_from_dir(self.__patients_paths)
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
        global_results = np.zeros(0)
        global_diagnosed_states = np.zeros(0)
        for i in range(len(self.__patients)):
            print(i, " iteration:")
            print("Split path...")
            training_paths, validation_paths, test_paths = self.__split_patients(test_number, validation_number)
            print("Generate Machine Learning model...")
            training_paths, validation_paths, test_paths = LeaveOneOutExperiment.__create_ml_dataset([training_paths,
                                                                                                      validation_paths,
                                                                                                      test_paths])
            feature_extractor = RHSDistanceExtract(MINIMUM_SAMPLES, SAMPLES)
            print("Training tensor extraction...")
            tensor_training, training_states, training_samples = feature_extractor.extract_rhs_known_state(training_paths)
            print("Training patients states: ", Counter(training_states))
            print("Validation tensor extraction...")
            tensor_validation, validation_states, validation_samples = feature_extractor.extract_rhs_known_state(validation_paths)
            print("Validation patients states: ", Counter(validation_states))
            print("Create learning model")
            self.__ml_model = MLModel(tensor_training, training_states, tensor_validation, validation_states)
            print("Testing model...")
            with open(path.join(SAVING_PATHS, "log_file_" + str(i) + ".txt"), 'w') as file:
                predicted_results = np.zeros(0)
                diagnosed_states = np.zeros(0)
                for test_path in test_paths:
                    id = FileManager.get_id_from_path(test_path)
                    state = FileManager.get_state_from_id(id)
                    test_tensor, states = feature_extractor.extract_rhs_file(test_path)
                    predicted_result, evaluation_result, avg_state = self.__ml_model.test_model(test_tensor, states)
                    file.write("File analized: " + test_path + " state predicted for file: " + str(avg_state)
                               + " state diagnosed: " + str(state) + " for patient: " + str(id) + "\n")
                    file.write("Evaluation result => loss: " + str(evaluation_result[0]) + ", accuracy: " + str(evaluation_result[1]) + "\n")
                    print("States: ", np.shape(states))
                    print("Result: ", np.shape(predicted_result))
                    predicted_results = np.concatenate((predicted_results, np.array(predicted_result)))
                    diagnosed_states = np.concatenate((diagnosed_states, np.array(states)))
                file.close()
        with open(path.join(SAVING_PATHS, "log_file.txt"), 'w') as file:
            accuracy, precision, recall, f_score = self.__ml_model.evaluate_results(predicted_results, diagnosed_states)
            file.write("GLOBAL ACCURACY: " + str(accuracy * 100) + "%\n")
            file.write("GLOBAL PRECISION: " + str(precision * 100) + "%\n")
            file.write("GLOBAL RECALL: " + str(recall * 100) + "%\n")
            file.write("GLOBAL F_SCORE: " + str(f_score * 100) + "%\n")
            file.close()

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

