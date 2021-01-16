# coding=utf-8

from Expreriment.RHSExperimentModel import Experiment
from DeepLearningClassifier.RHSDistanceExtraction import RHSDistanceExtract
from DeepLearningClassifier.TaskManager import TaskManager
from DatasetManager.FileManager import FileManager
from collections import Counter

NUM_FILE_SAMPLES = 50


class EmothawExperiment:

    """
        Nel costruttore dell'esperimento viene avviata la modellazione della rete neurale utilizzando i dati del dataset
        standard per addestramento, validazione e test.
    """
    def __init__(self, healthy_task, disease_task, test_task, file_samples):
        self.__healthy_task = healthy_task
        self.__disease_task = disease_task
        self.__test_task = test_task
        self.__file_samples = file_samples
        experiment = Experiment()
        experiment.execute_experiment("Dataset", self.__healthy_task, self.__disease_task, self.__test_task,
                                      self.__file_samples, "emothaw_experiment.txt")
        self.__ml_model = experiment.get_ml_model()

    """
        Questo metodo avvia il test del modello con i nuovi dati ottenuti dal dataset di emothaw.
    """
    def execute_emothaw_experiment(self):
        file_manager = FileManager("ConvertedEmothaw")
        rhs_extraction = RHSDistanceExtract(self.__file_samples, NUM_FILE_SAMPLES)
        ids_task = TaskManager.get_task_from_paths(file_manager.get_files_path(), self.__test_task)
        for id in ids_task:
            paths = ids_task.get(id)
            for task_path in paths:
                tensor = rhs_extraction.extract_rhs_from_unknown(task_path)
                result = self.__ml_model.predict_result(tensor)
                counter_result = Counter(result)
                print("Id: ", id, "file: ", task_path)
                healthy = counter_result.get(0) / len(result) * 100
                print("Healthy: ", healthy, "%")
                print("Disease: ", 100 - healthy, "%")
