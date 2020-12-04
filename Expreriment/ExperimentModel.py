# coding=utf-8

from os import path

from DeepLearningClassifier.FileManager import FileManager
from DeepLearningClassifier.MachineLearningModel import MLModel
from DeepLearningClassifier.RHSExtraction import RHSDistanceExtract
from DeepLearningClassifier.TaskManager import TaskManager

NUM_FILE_SAMPLES = 50
SAMPLES_LENGTH = 100
FEATURES_NUM = 3
INTERVALS_NUM = 3
MINIMUM_SAMPLES = 2500

TRAINING_FILE = 40
VALIDATION_FILE = 12
TEST_FILE = 10
MINIMUM_ROW_FILE = 2501


class Experiment:

    @staticmethod
    def execute_experiment(dataset, healthy_tasks, disease_tasks, test_tasks, log_file):
        file_manager = FileManager(dataset)
        file_paths = file_manager.get_files_path()
        print("Healthy task: ", healthy_tasks)
        print("Disease task: ", disease_tasks)
        print("Test task: ", test_tasks)
        training_list_disease, training_list_healthy, test_list_healthy, test_list_diseases, \
            validation_list_diseased, validation_list_healthy = TaskManager.split(file_paths, healthy_tasks, disease_tasks, test_tasks)
        model = [training_list_disease, training_list_healthy, test_list_diseases, test_list_healthy, validation_list_diseased, validation_list_healthy]
        model = TaskManager.check_file_dimension(TRAINING_FILE, VALIDATION_FILE, TEST_FILE, MINIMUM_ROW_FILE, model)
        print("Training file number: ", len(model[0]) + len(model[1]))
        print("Validation file number: ", len(model[3]) + len(model[5]))
        print("Test file number: ", len(model[2]) + len(model[3]))
        feature_extraction = RHSDistanceExtract(NUM_FILE_SAMPLES, SAMPLES_LENGTH, INTERVALS_NUM, FEATURES_NUM)
        print("Training tensor extraction...")
        tensor_training, states_training, samples_training, samples_file_training = \
            feature_extraction.extract_rhs_known_state(model[0] + model[1])
        print("Test tensor extraction...")
        tensor_test, states_test, samples_test, samples_file_test = \
            feature_extraction.extract_rhs_known_state(model[2] + model[3])
        print("Validation tensor extraction...")
        tensor_validation, states_validation, samples_validation, samples_file_validation = \
            feature_extraction.extract_rhs_known_state(model[4] + model[5])
        print("Create learning model...")
        ml_model = MLModel(tensor_training, states_training, tensor_validation, states_validation)
        ml_model.show_summary_graph()
        print("Testing model...")
        states_predicted, predicted_results = ml_model.test_model(tensor_test)
        evaluation_result, test_accuracy, test_precision, test_recall, test_f_score, wrong_classified, accuracy_file, \
            precision_file, recall_file, f1_score_file, wrong_paths = ml_model.classify_results(tensor_test, states_test,
                                                                                                predicted_results, states_predicted,
                                                                                                test_list_healthy + test_list_diseases, samples_file_test)
        print("Saving result...")
        save_file_path = path.join("experiment_result", log_file)
        FileManager.log_results(accuracy_file, evaluation_result, f1_score_file, precision_file, recall_file, save_file_path,
                                test_accuracy, test_f_score, test_precision, test_recall, wrong_classified, wrong_paths)
