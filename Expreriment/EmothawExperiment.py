# coding=utf-8

from collections import Counter

from DatasetManager.HandManager import HandManager
from DatasetManager.TaskManager import TaskManager
from DeepLearningClassifier.RHSDistanceExtraction import RHSDistanceExtract
from Expreriment.Experiment_1 import Experiment1

NUM_FILE_SAMPLES = 50


class EmothawExperiment:

    """
        Nel costruttore dell'esperimento viene avviata la modellazione della rete neurale utilizzando i dati del dataset
        standard per addestramento, validazione e src.
    """
    def __init__(self, healthy_task, disease_task, test_task, file_samples):
        self.__healthy_task = healthy_task
        self.__disease_task = disease_task
        self.__test_task = test_task
        self.__file_samples = file_samples
        experiment = Experiment1()
        experiment.execute_experiment("Dataset", self.__healthy_task, self.__disease_task, self.__test_task,
                                      self.__file_samples, "emothaw_experiment.txt")
        self.__ml_model = experiment.get_ml_model()

    """
        Questo metodo avvia il src del modello con i nuovi dati ottenuti dal dataset di emothaw.
    """
    def execute_emothaw_experiment(self):
        file_manager = HandManager("ConvertedEmothaw")
        rhs_extraction = RHSDistanceExtract(self.__file_samples, NUM_FILE_SAMPLES)
        ids_task = TaskManager.get_task_from_paths(file_manager.get_files_path(), self.__test_task)
        for id in ids_task:
            paths = ids_task.get(id)
            for task_path in paths:
                tensor = rhs_extraction.extract_rhs_file(task_path)
                result = self.__ml_model.predict_result(tensor)
                counter_result = Counter(result)
                print("Id: ", id, "file: ", task_path)
                healthy = counter_result.get(0) / len(result) * 100
                print("Healthy: ", healthy, "%")
                print("Disease: ", 100 - healthy, "%")
