# coding=utf-8

from DatasetManager.FileManager import FileManager
from DeepLearningClassifier.MachineLearningModel import MLModel
from DeepLearningClassifier.RTPExtraction import RTPExtraction
from DeepLearningClassifier.TaskManager import TaskManager

NUM_FILE_SAMPLES = 50

TRAINING_FILE = 40
VALIDATION_FILE = 12
TEST_FILE = 10


class Experiment:

    def __init__(self):
        self.__ml_model = None

    def execute_experiment(self, dataset, healthy_tasks, disease_tasks, test_tasks, minimum_samples, log_file):
        file_manager = FileManager(dataset)
        file_paths = file_manager.get_files_path()
        minimum_row_file = minimum_samples + 1
        print("Healthy tasks: ", healthy_tasks)
        print("Disease tasks: ", disease_tasks)
        print("Test tasks: ", test_tasks)
        # Separo le i tasks in base agli utenti che li hanno svolti.
        training_list_disease, training_list_healthy, test_list_healthy, test_list_diseases, \
            validation_list_diseased, validation_list_healthy = TaskManager.split(file_paths, healthy_tasks,
                                                                                  disease_tasks, test_tasks, training_number=25, validation_number=8)
        model = [training_list_disease, training_list_healthy, test_list_diseases, test_list_healthy,
                 validation_list_diseased, validation_list_healthy]
        # Elimino i file che non raggiungono la grandezza richiesta
        model = TaskManager.check_file_dimension(TRAINING_FILE, VALIDATION_FILE, TEST_FILE, minimum_row_file, model)
        print("After filtering")
        print("Training file number: ", len(model[0]) + len(model[1]))
        print("Validation file number: ", len(model[3]) + len(model[5]))
        print("Test file number: ", len(model[2]) + len(model[3]))
        feature_extraction = RTPExtraction(NUM_FILE_SAMPLES, minimum_samples)
        print("Training tensor extraction...")
        tensor_training, states_training, samples_training, samples_file_training = \
            feature_extraction.extract_rtp_known_state(model[0] + model[1])
        print("Test tensor extraction...")
        tensor_test, states_test, samples_test, samples_file_test = \
            feature_extraction.extract_rtp_known_state(model[2] + model[3])
        print("Validation tensor extraction...")
        tensor_validation, states_validation, samples_validation, samples_file_validation = \
            feature_extraction.extract_rtp_known_state(model[4] + model[5])
        print("Create learning model...")
        self.__ml_model = MLModel(tensor_training, states_training, tensor_validation, states_validation)
        self.__ml_model.show_summary_graph()
        print("Testing model...")

