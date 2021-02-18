# coding=utf-8

import os
import csv

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


class FeaturesManager:

    def __init__(self):
        FileManager.set_root_directory()
        if not os.path.exists(RESOURCE_DIR):
            os.mkdir(RESOURCE_DIR)
        self.__file_manager = FileManager("Dataset")
        self.__patients_path = self.__file_manager.get_patient_paths()
        self.__patients = FileManager.get_ids_from_dir(self.__patients_path)

    def create_features_file(self):
        paths = self.__file_manager.get_files_path()
        for task in TASKS:
            print("Extracting feature for task: ", TASKS_MAME.get(task))
            for id in self.__patients:
                task_file = FileManager.get_all_files_ids_tasks(id, task, paths)[0]
                print(task_file)
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
                user_dict = FeatureExtraction.get_features_for_task(dataset)
                # TODO aggiornare il dizionario

    @staticmethod
    def __read_row_data(file):
        row_data = []
        with open(file, 'r') as file:
            csv_file = csv.reader(file, quotechar="'", quoting=csv.QUOTE_NONNUMERIC, delimiter=' ')
            for row in csv_file:
                row_data.append(row)
            file.close()
        return row_data
