# coding=utf-8

import os
import csv

from DatasetManager import FileManager
from DatasetManager.Costants import *

RESOURCE_DIR = os.path.join("resource", "features")

PATHOLOGICAL_FILE_STRUCTURE = {
    'axis': 0,
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
        for i in range(len(self.__patients)):
            save_dir = os.path.join(RESOURCE_DIR, str(self.__patients[i]))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            task_files = FileManager.get_files_from_path(self.__patients_path[i], files=None)
            for file in task_files:
                task = FileManager.get_task_from_path(file)
                row_data = FeaturesManager.__read_row_data(file)

    @staticmethod
    def __read_row_data(file):
        row_data = []
        with open(file, 'r') as file:
            csv_file = csv.reader(file, quotechar="'", quoting=csv.QUOTE_NONNUMERIC, delimiter=' ')
            for row in csv_file:
                row_data.append(row)
            file.close()
        return row_data

    @staticmethod
    def get_other_status(status):
        return 0 if status == 1 else 1