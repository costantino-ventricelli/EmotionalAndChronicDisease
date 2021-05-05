# coding=utf-8

import os

import numpy as np
import pandas as pd

from DatasetManager import HandManager
from DatasetManager import TaskManager
from DatasetManager.Costants import *

FEATURES_DIRECTORY = os.path.join("resource", "features")

"""
    Questa classe permette di creare un dataset sotto forma di dizionario così da impedire che durante le fasi di 
    addestramento e validazione all'interno dei dati di input per i modelli finiscano dati che dovrebbero essere utilizzati
    solo in fase di test.
    la struttura che si crea è la seguente:
    
    {key->[id_paziente] : value-> ground_thought: valore di verità,
                                 tasks{ key->[task] : value->feature estratte per quel task}
                                 }
"""


class CreateDictDataset:

    """
        L'inizializzazione della classe prende in input la lista delle feature da selezionare all'interno del dataset,
        ovviamente la lista può derivare sia da un random forest sulle feature che da una selezione manuale delle stesse.
        Inoltre in input vengono dati i tasks per l'addestramento degli utenti sani e i tasks per l'addestramento degli
        utenti malati. Se i task non vengono forniti in fase di inizializzazione verranno considerati tutti per entrambe le
        categorie.
    """
    def __init__(self, features_selected, healthy_task, disease_task):
        self.__patients_list = CreateDictDataset.__read_patient_list()
        self.__dataset_dict = {}
        self.__features_selected = features_selected
        for id in self.__patients_list:
            self.__dataset_dict[id] = {
                'ground_thought': HandManager.get_state_from_id(id),
                'user_tasks': dict.fromkeys(TASKS, np.ndarray)
            }
        if healthy_task is None:
            healthy_task = TASKS
        if disease_task is None:
            disease_task = TASKS
        self.__healthy_task = healthy_task
        self.__disease_task = disease_task
        # Qui inizia la scansione task per task la quale permette di prelevare le feature estratte per ogni utente
        for task in TASKS:
            print("Opening file: ", TASKS_MAME.get(task) + '.csv')
            with open(os.path.join(FEATURES_DIRECTORY, TASKS_MAME.get(task) + '.csv'), 'r') as file:
                # Il file csv viene letto in un dataframe pandas per gestirne l'impressionate mole.
                dataframe = pd.read_csv(file)
                # Ottengo gli header dal file.
                features = dataframe.columns.to_list()
                for i in range(len(self.__patients_list)):
                    # Dopo di che la scansione delle righe accoppierà ogni riga al paziente corretto tramite il file che
                    # viene generato all'estrazione delle feature che mantiene l'ordinamento che hanno avuto i pazienti
                    # al momento dell'estrazione
                    row = np.zeros(0)
                    # Ogni riga del data frame contiene le feature estratte per l'utente.
                    row_data = dataframe.iloc[i]
                    # Adesso mi muovo nella riga selezionata prendendo solo le colonne che compaiono nelle feature selez
                    # onate per la classificazione.
                    for index in self.__features_selected:
                        # Se il valore nel dataframe è una tringa è sicuro che sia un numero complesso memorizzato nella
                        # forma (#.###+j###.##) quindi questa verifica mi permette di trasformare la stringa in un numero
                        # complesso così che possa essere gestita.
                        if isinstance(row_data[features.index(index)], str):
                            row = np.append(row, np.abs(eval(row_data[features.index(index)])))
                        else:
                            row = np.append(row, row_data[features.index(index)])
                    row = np.array(row).astype(np.float32)
                    # Questi tre where mi permettono di pulire il dataset dai valori che hanno generato errori di calcolo
                    # durante l'estrazione delle feature.
                    row = np.where(row == np.inf, np.finfo(np.float32).max, row)
                    row = np.where(row == -np.inf, np.finfo(np.float32).min, row)
                    row = np.where(np.isnan(row), 0.00, row)
                    # Aggiorno il dizionario
                    user_tasks = self.__dataset_dict.get(self.__patients_list[i]).get('user_tasks')
                    user_tasks[task] = row
                file.close()

    """
        Questo metodo permette di ottenere le matrici dei dataset di training e test assieme ai loro valori di verità.
        Passando l'id per il paziente di test e i relativi task il metodo restituisce il dataset di training senza i file
        relativi al paziente di test.
    """
    def get_dataset(self, test_id, test_task=None):
        # Se non viene passato alcun valore per i i task di test vengono considerati come task di test tutti i restanti
        # del dataset.
        if test_task is None:
            test_task = TaskManager.get_tasks_difference(TASKS, self.__healthy_task, self.__disease_task)
        healthy_dataset = []
        disease_dataset = []
        test_dataset = []
        test_ground_thought = []
        for key, value in self.__dataset_dict.items():
            # Scansiono tutto il dizionario individuando l'id del test per e l resto per il dataset di trainig
            # differenziando tra utenti sani e malati in modo da poter poi bilanciare il dataset.
            user_tasks = value.get('user_tasks')
            if key == test_id:
                test_dataset = CreateDictDataset.__get_task_from_patient(test_task, test_dataset, user_tasks)
                test_ground_thought = [value.get('ground_thought') for _ in range(len(test_dataset))]
            else:
                ground_thought = value.get('ground_thought')
                if ground_thought == HEALTHY:
                    healthy_dataset = CreateDictDataset.__get_task_from_patient(self.__healthy_task, healthy_dataset, user_tasks)
                else:
                    disease_dataset = CreateDictDataset.__get_task_from_patient(self.__disease_task, disease_dataset, user_tasks)
        healthy_dataset, disease_dataset = HandManager.balance_dataset(healthy_dataset, disease_dataset)
        train_ground_thought = [HEALTHY for _ in range(len(healthy_dataset))] + [DISEASE for _ in range(len(disease_dataset))]
        return healthy_dataset + disease_dataset, test_dataset, train_ground_thought, test_ground_thought, test_task

    def get_patient_list(self):
        return self.__patients_list

    """
        Questo metodo permette di ottenre la matrice del dataset passato in input aggiungendo i task specificati nel primo
        parametro dall'insieme dei task presenti per l'utente selezionato.
    """
    @staticmethod
    def __get_task_from_patient(tasks, dataset, user_tasks):
        for sub_task in tasks:
            if not isinstance(sub_task, list):
                sub_task = [sub_task]
                for task in sub_task:
                    row = user_tasks.get(task)
                    if not isinstance(row, type) and np.any(row):
                        dataset.append(row)
        return dataset

    @staticmethod
    def __read_patient_list():
        with open(os.path.join(FEATURES_DIRECTORY, "patients.txt")) as file:
            data = file.readline().replace('[', '').replace(']', '')
            results = np.array(data.split(',')).astype(int)
            file.close()
        return results
