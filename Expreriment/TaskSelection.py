# coding=utf-8

import numpy as np

from DatasetManager import FileManager
from DatasetManager.Costants import *
from DeepLearningClassifier import *
from copy import deepcopy


KEY_TUPLE = 0
VALUE_TUPLE = 1


class TaskSelection:

    """
        @:param samples_len: contiene la lunghezza che avranno i campioni all'interno dei tensori
        @:param minumum_samples: contiene la lunghezza minima in termini di righe dei file da considerare come adatti
        Il metodo di init permette di generare una prima classifica dei task che ottengono i risultati migliori basandosi
        sulle quattro metriche messe in ordine di importanza all'interno della tupla (accuracy, precision, recall, f_score)
    """
    def __init__(self, samples_len, minimum_samples):
        # Lista dei task del dataset hand
        self.__tasks = [CLOCK, NATURAL_SENTENCE, PENTAGON, MATRIX_1, MATRIX_2, MATRIX_3, TRIAL_1, T_TRIAL_1, T_TRIAL_2,
                        TRIAL_2, HELLO, V_POINT, H_POINT, SQUARE, SIGNATURE_1, SIGNATURE_2, COPY_SPIRAL, TRACED_SPIRAL,
                        BANK_CHECK, LE, MOM, WINDOW, LISTENING]
        self.__samples_len = samples_len
        self.__minimum_samples = minimum_samples
        self.__best_results = {}
        self.__file_manager = FileManager("Dataset")
        self.__feature_extraction = RHSDistanceExtract(minimum_samples, samples_len)
        # Con questo loop posso avviare apprendimenti su ogni task del dataset, ciò mi permetterà di individiare il task
        # migliore da cui iniziare la selezione.
        for task in self.__tasks:
            # Per il test vengono selezionati due utenti uno etichettato come sano e uno etichettato come malato, per ottenere
            # misuarizioni sulle prestazioni del sistema più precise ed affidabili.
            paths, test_paths, validation_number = self.__select_paths_from_tasks([task])
            # Questo controllo evita la catastrofe in quanto permette di verificare che ci siano entrambe le classi per
            # poter effettuare la classificazione, quindi con questo controllo le possibilità di incorrere in problemi di
            # bilanciamento del dataset si riduce accorpandosi con il controllo fatto al momento dell'estrazione delle feature.
            if len(test_paths) > 1:
                # Estraggo i segmenti RHS per training, validazione e test del sistema.
                test_states, test_tensor, training_states, training_tensor, validation_states, validation_tensor = self.__extract_rhs_segment(
                    paths, test_paths, validation_number)
                # Genero il sistema e valuto i risultati.
                self.__best_results = TaskSelection.__create_and_evaluate_model(task, test_states, test_tensor, training_states, training_tensor,
                                                                                validation_states, validation_tensor, self.__best_results)
            else:
                print("The dataset is not balanced, there aren't the needed class for the classification.")

    """
        Questo metodo mi permette di avviare la selezione dei task partendo da quelli che hanno dato il miglior risultato
        nell inizializzazione.
    """
    def execute_task_selection(self):
        # Selezioni i task migliori con i rispettivi risultati.
        previous_max = max(self.__best_results.items())
        actual_tuple = max(self.__best_results.items())
        # La selezione continuerà fino al deterioramento dei risultati, ovvero appena uno dei quatto paramentri si abbassa
        # la selezione viene interrotta.
        while previous_max[KEY_TUPLE] <= actual_tuple[KEY_TUPLE]:
            # Seleziono la lista dei task appartenente al migliore dei risultati selezionato precedentemente.
            actual_tasks = actual_tuple[VALUE_TUPLE]
            # Creo il nuovo dizionario che permetterà di salvare i risultati
            best_results = {}
            # Scansionando tutti i task dovrei essere in grado di aggiungere nuovi task alla selezione.
            for task in self.__tasks:
                # Verifico che il task selezionato non sia già stato preso in analisi.
                if task not in actual_tasks:
                    # Aggiungo il task alla lista di nuovi task da analizzare.
                    actual_tasks.append(task)
                    # Seleziono i file per il modello compresi i file di test e il numero di file che verranno destinati
                    # alla validazione.
                    paths, test_paths, validation_number = self.__select_paths_from_tasks(actual_tasks)
                    # Estraggo i segmenti RHS per i tensori.
                    test_states, test_tensor, training_states, training_tensor, validation_states, validation_tensor = self.__extract_rhs_segment(
                        paths, test_paths, validation_number)
                    # Addestro e valuto il modello
                    best_results = TaskSelection.__create_and_evaluate_model(actual_tasks, test_states, test_tensor, training_states, training_tensor,
                                                                             validation_states, validation_tensor, best_results)
                # Salvo i risultati precedenti prima di aggiornare il sitema con i nuovi dati.
                previous_max = deepcopy(actual_tuple)
                actual_tuple = max(best_results.items())
        return previous_max

    def __select_paths_from_tasks(self, tasks):
        healthy_ids, disease_ids = FileManager.get_healthy_disease_list()
        paths = []
        for task in tasks:
            paths += TaskManager.get_list_from_task(task, self.__file_manager.get_files_path())
        paths = FileManager.filter_file(paths, min_dim=self.__minimum_samples + 1)
        healthy_paths = TaskSelection.__get_test_for_state(healthy_ids, paths)
        disease_paths = TaskSelection.__get_test_for_state(disease_ids, paths)
        if not healthy_paths or not disease_paths:
            test_paths = []
        else:
            test_paths = healthy_paths + disease_paths
        for i in range(len(test_paths)):
            if test_paths[i] in paths:
                del paths[paths.index(test_paths[i])]
        validation_number = int(np.ceil(len(paths) * 0.2))
        return paths, test_paths, validation_number

    @staticmethod
    def __get_test_for_state(ids, paths):
        i = 0
        return_paths = []
        while len(return_paths) != 0 and i < len(return_paths):
            return_paths = FileManager.get_all_file_of_id(ids[i], paths)
            i += 1
        return return_paths

    """
    
    """
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

