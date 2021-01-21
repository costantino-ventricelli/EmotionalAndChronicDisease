# coding=utf-8

import os
import re

import pandas

RESOURCE_DIRECTORY = "resource"


class FileManager:

    def __init__(self, dataset_name):
        working_dir = os.getcwd()
        split_dir = working_dir.split(os.sep)
        if split_dir[len(split_dir) - 1] == "src":
            split_dir.pop()
        working_dir = FileManager.__join_directory(split_dir)
        os.chdir(working_dir)
        self.__dataset = os.path.join(RESOURCE_DIRECTORY, dataset_name)
        self.__dataset_directory = FileManager.get_path_directories(self.__dataset)
        self.__patient_paths = FileManager.__get_patient_paths(self.__dataset_directory)
        self.__files_path = FileManager.__get_files_from_paths(self.__patient_paths)

    def get_dataset_directory(self): return self.__dataset_directory

    def get_patient_paths(self): return self.__patient_paths

    def get_files_path(self): return self.__files_path

    @staticmethod
    def __join_directory(split_dir):
        path = os.sep
        for slice in split_dir:
            path = os.path.join(path, slice)
        return path

    """
        @:param id: contiene l'id da verificare.
        @:healthy_id: contiene la lista degli id di cui verificare l'appartenenza.
        @:return: 0 se l'id appartiene alla lista di utenti sani
                  1 se l'id appartiene alla lista di utenti malati.
    """
    @staticmethod
    def get_state_from_id(id, healthy_ids):
        if id in healthy_ids:
            return 0
        else:
            return 1

    """
        @:param main_directory_path: contiene il path per la directory generale del dataset
        @:return: restituisce tutte le directory presenti nel dataset
    """
    @staticmethod
    def get_path_directories(main_directory_path):
        directories = []
        try:
            # Itero su tutte le cartelle contenute nella directory principale del dataset.
            for directory in os.listdir(main_directory_path):
                if not (directory.endswith(".DS_Store")):
                    # Genero i percorsi per ogni directory
                    directories.append(os.path.join(main_directory_path, directory))
        except OSError as er:
            raise er
        return directories

    """
        @:param directories: Contiene la lista delle directories persenti nella cartella principale del dataset.
        @:return: restituisce i percorsi di tutti i pazienti contenuti nelle directories superiori del dataset.
    """
    @staticmethod
    def __get_patient_paths(directories):
        patients = []
        try:
            # Itero su tutte le sotto cartelle del dataset.
            for directory in directories:
                for patient in os.listdir(directory):
                    if not patient.endswith(".DS_Store"):
                        # Genero il percorso che condurrà ad ogni directory che rappresenta i pazienti nel dataset.
                        patients.append(os.path.join(directory, patient))
        except OSError as er:
            raise er
        return patients

    """
        @:param patients_directory: contiene tutti percorsi delle directories che rappresentano i pazienti nel dataset.
        @:return: restituisce tutti i percorsi del dataset con tutti i file per i task in esso contenuti.
    """
    @staticmethod
    def __get_files_from_paths(patients_directory):
        files = []
        try:
            # Itero su tutte le directory del dataset prelevando tutti i file di task con estensione txt.
            for patient in patients_directory:
                files = FileManager.get_files_from_path(patient, files)
        except OSError as er:
            raise er
        return files

    """
        @:param directory: contiene la cartella da analizzare per ottenere la lista dei file.
        @:param files: contiene la lista di file eventualmente già prelevati, se impostata a None genera e restituisce 
                la lista dei file solo della cartella indicata.
        @:return: restituisce una lista di percorsi aggiornata con i nuovi percorsi individuati nella cartella.
    """
    @staticmethod
    def get_files_from_path(directory, files):
        if files is None:
            files = []
        try:
            for file in os.listdir(directory):
                if file.endswith(".txt"):
                    files.append(os.path.join(directory, file))
        except OSError as er:
            raise er
        return files

    """
        @:param path: contiene il percorso di un file di task
        @:return: restituisce l'id del paziente che ha svolto quel task.
    """
    @staticmethod
    def get_id_from_path(path):
        # Viene effettuata la ricerca dell'espressione regolare che permetterà di prelevare l'id del paziente dal percorso.
        return int(re.search(r'_u(.*?)_', path).group(1))

    @staticmethod
    def get_ids_from_paths(paths):
        ids = []
        for path in paths:
            split = path.split(os.sep)
            ids.append(int(split[len(split) - 1]))
        return ids

    """
        @:param path: contiene il percorso di un task.
        @:return: restituisce l'dentificativo del task che quel file rappresenta
    """
    @staticmethod
    def get_task_from_path(path):
        # Ricerca l'espressione regolare che identifica il task
        return re.search(r'_u(.*?)_(.*?).txt', path).group(2)

    """
        @:return: il metodo restituisce due liste di id, una per gli utenti identificati dallo screening iniziale come 
            sane e una per gli utenti identificati come malati.
    """
    @staticmethod
    def get_healthy_disease_list():
        column_names = ["ID", "DEMENTIA"]
        diagnosis_table_path = os.path.join(RESOURCE_DIRECTORY, "diagnosis_table.csv")
        # Il metodo apre il file contenente i dati per ogni utente che identificano con 0 un utente sano e con 1 un utente
        # malato.
        file = pandas.read_csv(diagnosis_table_path, sep=";", header=None, names=column_names)
        # Trasformo in due liste le informazioni contenute nel file, una per gli ids
        ids = file["ID"].tolist()
        # E una che associerà ogni id allo stato di salute.
        dementias = file["DEMENTIA"].tolist()
        healthy_ids = []
        disease_ids = []
        # Per ogni elemento delle liste verifico e smisto i gli ids nelle liste di appartenenza.
        for i in range(len(dementias)):
            healthy_ids.append(ids[i]) if dementias[i] == 0 else disease_ids.append(ids[i])
        return healthy_ids, disease_ids

    """
        @:param x_axis: è la lista dei punti campionati per l'asse x
        @:param y_axis: è la lista dei punti campionati per l'asse y
        @:param time_stamp: è la lista dei punti campionati per il timestamp
        @:param pen_status: è la lista dei punti campionati come bottom status nella scrittura dei task.
        @:return: il metodo restituice una tupla di liste contentente tutte le liste di input private di tutti i punti 
            considerati come doppione.
    """
    @staticmethod
    def delete_duplicates(x_axis, y_axis, time_stamp, bottom_status):
        # Genero le nuove liste per la restituzione dei dati
        x_axis_new = [x_axis[0]]
        y_axis_new = [y_axis[0]]
        time_stamp_new = [time_stamp[0]]
        bottom_status_new = [bottom_status[0]]
        # Itero per tutti i campioni nelle liste e considero doppione ognuno dei campioni che presenta un timestamp
        # identico a quello del campione precedente.
        for i in range(1, len(x_axis)):
            if time_stamp[i] != time_stamp[i - 1]:
                x_axis_new.append(x_axis[i])
                y_axis_new.append(y_axis[i])
                time_stamp_new.append(time_stamp[i])
                bottom_status_new.append(bottom_status[i])
        return x_axis_new, y_axis_new, time_stamp_new, bottom_status_new

    """
        @:param path: percorso del file che il metodo analizza.
        @:return: restituisco il numero di righe che compongono il file.
    """
    @staticmethod
    def get_file_rows(path):
        with open(path, 'r') as file:
            length = len(list(file))
            file.close()
        return length

    """
        @:param paths: contiene la lista dei percorsi che indicano i file da filtrare.
        @:param min_dim: indica il numero minimo di righe che il file deve avere.
        @:return: restituisce una nuova lista di file filtrata in base al numero minimo di righe.
    """
    @staticmethod
    def filter_file(paths, min_dim):
        filtered_paths = []
        for path in paths:
            if FileManager.get_file_rows(path) >= min_dim:
                filtered_paths.append(path)
        return filtered_paths

    @staticmethod
    def log_results(accuracy_file, evaluation_result, f1_score_file, precision_file, recall_file, save_file_path,
                    test_accuracy, test_f_score, test_precision, test_recall, wrong_classified, wrong_paths):
        with open(save_file_path, 'w') as file:
            file.write("loss_value: " + str(evaluation_result[0]) + "\n")
            file.write("accuracy_value: " + str(evaluation_result[1]) + "\n")
            file.write("Test accuracy: " + str(test_accuracy) + "\n")
            file.write("Test precision: " + str(test_precision) + "\n")
            file.write("Test recall: " + str(test_recall) + "\n")
            file.write("Test f1_score: " + str(test_f_score) + "\n")
            file.write("Wrong file classified total: " + str(wrong_classified) + "\n")
            file.write("Accuracy file classified: " + str(accuracy_file) + "\n")
            file.write("Precision file classified: " + str(precision_file) + "\n")
            file.write("Recall file classified: " + str(recall_file) + "\n")
            file.write("F1 score file classified: " + str(f1_score_file) + "\n")
            file.write("Wrong file classified: \n")
            for path in wrong_paths:
                file.write("\t" + path + "\n")
            file.close()
