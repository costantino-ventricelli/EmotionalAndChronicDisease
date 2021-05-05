# coding=utf-8

import csv
import os

from DatasetManager import HandManager
from DeepLearningClassifier import *

METRICS_KEY = 0
HEALTHY_VALUE = 1
DISEASE_VALUE = 2
FEATURE = 3

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"


class Experiment3:

    """
        @:param prev_file: contiene il percorso del file frutto di un esecuzione precedente, nel momento in cui viene
                passato None il processo assume che è la prima volta che viene avviata l'analisi
        @:param tasks: contiene la lista di task selezionata.
        @:param samples_len: contiene la lunghezza di ogni campione.
        @:param minumum_samples: contiene il numero minimo di esempi che devono essere presenti nel file.
        @:param category: contiene il nome della categoria che si sta selezionando, il quale nel caso di prima selezione
                            diverrà il nome del file di salvataggio.
    """
    def __init__(self, prev_file, tasks, minimum_samples, samples_len, category):
        HandManager.set_root_directory()
        self.__tasks = tasks
        # Il file contente i valori calcolati ci permette di ricominciare l'esecuzione del programma nel caso dovesse
        # interrompersi per qualsiasi ragione.
        self.__prev_file = prev_file
        if prev_file is not None:
            self.__results = self.__set_previous_state()
        else:
            self.__results = {}
        self.__minimum_samples = minimum_samples
        self.__samples_len = samples_len
        self.__category = category
        # Ottengo l'intero dataset da HandManager
        self.__learning_method = DeepLeaveOneOut(self.__minimum_samples, FEATURE, RHSDistanceExtract(minimum_samples, samples_len), self.__samples_len, "Dataset")

    """
        Questo metodo avvia la selezione dei task per gli utenti etichettati come sani, per farlo richiama il metodo
        che permetterà di selezionare i task per gli utenti etichettati come non sani tante volte quanti sono i task
        che si è deciso di analizzare.
    """
    def start_healthy_selection(self):
        print("Starting selection on tasks: ", self.__tasks)
        for healthy_task in self.__tasks:
            self.__start_disease_selection(healthy_task)

    """
        In questo metodo ogni task selezionato per gli utenti sani viene combianto con tutti i task rimanenti.
        I quali vengono utilizzati per selezionare campioni rappresententanti gli utenti malati.
    """
    def __start_disease_selection(self, healthy_task):
        for disease_task in self.__tasks:
            result = self.__start_selection(healthy_task, disease_task)
            if result is not None:
                print("Execution ended: ", result.items())
            else:
                print("Combination already tried")

    """
        In questo metodo effettua una serie di controlli sui task prima di avviare la modellazione della rete con i task
        risultati idonei alla modellazione, in caso di successo il metodo restituisce un dizionario contentente come chiave
        le metriche calcolate per la modellazione e come valore la lista dei task che hanno generato quei valori.
    """
    def __start_selection(self, healthy_task, disease_task):
        # Prima di tutto verifico che i task selezionati siano differenti.
        if healthy_task != disease_task:
            # A questo punto con l'ausilio del file passato al costruttore escudo tutte le combinazioni che sono già
            # state provate.
            if not self.__is_already_do(healthy_task, disease_task):
                # Se la combinazione è nuova allora viene avviata la modellazione leave_one_out che restituirà le quattro
                # metriche che ci permetteranno di valutare i risultati.
                accuracy, precision, recall, f_score = self.__learning_method.leave_one_out(healthy_task, disease_task)
                print("Model evaluated")
                # A questo punto vengono eseguiti i passaggi necessari alla creazione della directory di salvataggio e
                # viene generato un nuovo file nel caso di prima esecuzione del codice, oppure vine aggiornato il file
                # esisitente.
                directory = os.path.join("experiment_result", "experiment_3")
                if not os.path.exists(directory):
                    os.mkdir(directory)
                if self.__prev_file is None:
                    self.__prev_file = os.path.join(directory, self.__category + ".txt")
                    file = open(self.__prev_file, 'w')
                else:
                    file = open(self.__prev_file, 'a')
                csv_file = csv.writer(file, delimiter=';')
                csv_file.writerow([(accuracy, precision, recall, f_score), healthy_task, disease_task])
                file.close()
                return {(accuracy, precision, recall, f_score): [healthy_task, disease_task]}
        return None

    """
        Questo metodo verifica se una combinazione di task è già stata provata, verificando la sua esistenza nel diziona-
        rio di istanza contenente i risultati della valutazione.
    """
    def __is_already_do(self, healthy_task, disease_task):
        for _, items in self.__results.items():
            for item in items:
                if item[HEALTHY_STRING].strip() == healthy_task.strip() and item[DISEASE_STRING].strip() == disease_task.strip():
                    return True
        return False

    """
        Questo metodo ha il compito di reimpostare i valori del dizionaro sulla base del file passato al costruttore.
    """
    def __set_previous_state(self):
        with open(self.__prev_file, 'r') as file:
            csv_file = csv.reader(file, delimiter=';')
            result = {}
            for row in csv_file:
                key = eval(row[METRICS_KEY])
                value_dict = {HEALTHY_STRING: row[HEALTHY_VALUE], DISEASE_STRING: row[DISEASE_VALUE]}
                if key in result.keys():
                    list_of_dict = result.get(key)
                    list_of_dict.append(value_dict)
                else:
                    result[key] = [value_dict]
            file.close()
        return result
