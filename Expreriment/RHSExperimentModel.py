# coding=utf-8

from os import path

from DatasetManager import FileManager
from DeepLearningClassifier import *

NUM_FILE_SAMPLES = 50

TRAINING_FILE = 40
VALIDATION_FILE = 12
TEST_FILE = 10


class Experiment:

    def __init__(self):
        self.__ml_model = None

    def execute_experiment(self, dataset, healthy_tasks, disease_tasks, test_tasks, minimum_samples, log_file):
        log_file = path.join("experiment_result", log_file)
        file_manager = FileManager(dataset)
        file_paths = file_manager.get_files_path()
        minimum_row_file = minimum_samples + 1
        print("Healthy task: ", healthy_tasks)
        print("Disease task: ", disease_tasks)
        print("Test task: ", test_tasks)
        # Separo le i task in base agli utenti che li hanno svolti.
        training_list_disease, training_list_healthy, test_list_healthy, test_list_diseases, \
            validation_list_diseased, validation_list_healthy = TaskManager.split(file_paths, healthy_tasks, disease_tasks, test_tasks, training_number=25, validation_number=8)
        # Elimino i file che non raggiungono la grandezza richiesta
        training_list_disease, training_list_healthy, test_list_diseases, test_list_healthy, \
            validation_list_diseased, validation_list_healthy = \
            TaskManager.check_file_dimension(TRAINING_FILE, VALIDATION_FILE, TEST_FILE, minimum_row_file,
                                             [training_list_disease, training_list_healthy, test_list_diseases,
                                              test_list_healthy, validation_list_diseased, validation_list_healthy])
        print("After filtering")
        print("Training file number: ", len(training_list_healthy) + len(training_list_disease))
        print("Validation file number: ", len(validation_list_healthy) + len(validation_list_diseased))
        print("Test file number: ", len(test_list_healthy) + len(test_list_diseases))
        # Estraggo i campioni RHS dalle diverse liste di punti campionati.
        feature_extraction = RHSDistanceExtract(minimum_samples, NUM_FILE_SAMPLES)
        print("Training tensor extraction...")
        tensor_training, states_training, samples_training = \
            feature_extraction.extract_rhs_known_state(training_list_disease + training_list_healthy)
        print("Validation tensor extraction...")
        tensor_validation, states_validation, samples_validation = \
            feature_extraction.extract_rhs_known_state(validation_list_diseased + validation_list_healthy)
        # Genero il modello di DeepLearning con i tensori genererati.
        print("Create learning model...")
        self.__ml_model = MLModel(tensor_training, states_training, tensor_validation, states_validation)
        self.__ml_model.show_summary_graph()
        print("Testing model...")
        test_paths = test_list_diseases + test_list_healthy
        predicted_results = []
        theoretical_results = []
        with open(log_file, 'w') as file:
            for test_path in test_paths:
                id = FileManager.get_id_from_path(test_path)
                state = FileManager.get_state_from_id(id)
                file.write("Test for patient: " + str(id) + "\n")
                file.write("Theoretical state for patient: " + str(state) + "\n")
                tensor, theoretical_result = feature_extraction.extract_rhs_file(test_path)
                predicted_result, evaluation_result, sample_average = self.__ml_model.test_model(tensor, theoretical_result)
                file.write("Evaluation result  loss: " + str(evaluation_result[0]) + " accuracy: " + str(evaluation_result[1]) + "\n")
                file.write("State predicted: " + str(sample_average) + "\n")
                predicted_results.append(sample_average)
                theoretical_results.append(state)
                print("\n")
            accuracy, precision, recall, f_score = MLModel.evaluate_results(predicted_results, theoretical_results)
            file.write("Global accuracy: " + str(int(accuracy * 100)) + "%\n")
            file.write("Global precision: " + str(int(precision * 100)) + "%\n")
            file.write("Global recall: " + str(int(recall * 100)) + "%\n")
            file.write("Global f_score: " + str(int(f_score * 100)) + "%\n")
            file.close()

    def get_ml_model(self):
        return self.__ml_model
