# coding=utf-8

import os
import csv
import numpy as np

from DatasetManager import FileManager
from .FeatureExtraction import FeatureExtraction
from DatasetManager.Costants import *

RESOURCE_DIR = os.path.join("resource", "features")

TASK_STRUCTURE = {
    'x_axis': 0,
    'y_axis': 1,
    'pressure': 2,
    'timestamp': 3,
    'azimuth': 4,
    'altitude': 5,
    'pen_status': 6
}


"""
    Questa classe permette di ottenere tutte le feature dal dataset scansionando prima tutti i task di cui il dataset è
    composto, quindi basta utilizzare la classe di lettura del dataset corretto per ottenere le feature.
"""


class FeaturesManager:

    def __init__(self):
        FileManager.set_root_directory()
        if not os.path.exists(RESOURCE_DIR):
            os.mkdir(RESOURCE_DIR)
        self.__file_manager = FileManager("Dataset")
        self.__patients_path = self.__file_manager.get_patient_paths()
        self.__patients = FileManager.get_ids_from_dir(self.__patients_path)

    def create_features_file(self):
        with open(os.path.join(RESOURCE_DIR, "patients.txt"), 'w') as file:
            file.write(str(self.__patients))
            file.close()
        paths = self.__file_manager.get_files_path()
        for task in TASKS:
            print("Extracting feature for task: ", TASKS_MAME.get(task))
            task_dict = FeatureExtraction.get_file_dictionary()
            with open(os.path.join(RESOURCE_DIR, (TASKS_MAME.get(task) + '.csv')), 'w') as file:
                csv_file = csv.DictWriter(file, fieldnames=task_dict.keys())
                csv_file.writeheader()
                file.close()
            for id in self.__patients:
                # Per ogni paziente si legge il file di task generando 6 array differenti ognuno dei quali comporrà il
                # dataset che verrà analizzato per estrarre le feature.
                task_file = FileManager.get_all_files_ids_tasks(id, task, paths)[0]
                rows_data = FeaturesManager.__read_row_data(task_file)
                x_point = []
                y_point = []
                pressure_point = []
                time_stamps = []
                azimuth_point = []
                altitude_point = []
                pen_status = []
                for row in rows_data:
                    x_point.append(row[TASK_STRUCTURE.get('x_axis')])
                    y_point.append(row[TASK_STRUCTURE.get('y_axis')])
                    pressure_point.append(row[TASK_STRUCTURE.get('pressure')])
                    time_stamps.append(row[TASK_STRUCTURE.get('timestamp')])
                    azimuth_point.append(row[TASK_STRUCTURE.get('azimuth')])
                    altitude_point.append(row[TASK_STRUCTURE.get('altitude')])
                    pen_status.append(row[TASK_STRUCTURE.get('pen_status')])
                dataset = [x_point, y_point, pressure_point, time_stamps, altitude_point, altitude_point, pen_status]
                # I risultati di estrazione vengono salvati in un dizionario che viene scritto all'interno del file che
                # contiene le feature estratte per tutti gli utenti.
                user_dict = FeatureExtraction.get_file_dictionary()
                try:
                    user_dict = FeatureExtraction.get_features_for_task(dataset)
                    for key, item in user_dict.items():
                        if isinstance(item, list):
                            item = item[0]
                        user_dict[key] = item
                except RuntimeError or RuntimeError as error:
                    if os.path.exists(os.path.join(RESOURCE_DIR, "error_log.log")):
                        mode = 'a'
                    else:
                        mode = 'w'
                    with open(os.path.join(RESOURCE_DIR, "error_log.log"), mode) as file:
                        file.write("DATASET LEN: " + str(np.shape(dataset)))
                        file.write("ID: " + str(id))
                        file.write("File: " + task_file)
                        file.close()
                with open(os.path.join(RESOURCE_DIR, (TASKS_MAME.get(task) + '.csv')), 'a') as file:
                    csv_file = csv.DictWriter(file, fieldnames=task_dict.keys())
                    csv_file.writerow(user_dict)
                    file.close()

    @staticmethod
    def __read_row_data(file):
        row_data = []
        with open(file, 'r') as file:
            csv_file = csv.reader(file, quotechar="'", quoting=csv.QUOTE_NONNUMERIC, delimiter=' ')
            for row in csv_file:
                row_data.append(row)
            file.close()
        return row_data
