# coding=utf-8

from copy import deepcopy
from .Features import Features
from DatasetManager.Costants import *

import numpy as np
import csv
import os
import re

TASK_STRUCTURE = {
    'x_axis': 0,
    'y_axis': 1,
    'pressure': 2,
    'timestamp': 3,
    'azimuth': 4,
    'altitude': 5,
    'pen_status': 6
}

HEADER_FILE = 'header_file.csv'
FEATURE_FILE_LIST = 'feature_list.csv'


#TODO: Adesso bisogna si finire di mettere l'estrazione di tutte le feature, ma disogna individuare un modo per generare
# i file, probabilmente conviene invertire l'ordine di scansione, quindi i task che formano un file con tutti gli utenti.
class FeatureExtraction:

    @staticmethod
    def get_features_for_task(dataset):
        features = FeatureExtraction.__get_feature_list()
        for feature in features:
            result = FeatureExtraction.__get_feature_command(feature, dataset)

    @staticmethod
    def __get_statistic_value(data):
        return {
            '[mean]': np.mean(data),
            '[median]': np.median(data),
            '[stan. dev.]': np.std(data),
            '[1 per]': np.percentile(data, 10),
            '[99 per]': np.percentile(data, 99),
            '[1-99 per]': np.percentile(data, 99) - np.percentile(data, 10)
        }

    @staticmethod
    def __get_feature_list():
        feature = []
        with open(os.path.join("resource", FEATURE_FILE_LIST), 'r') as file:
            csv_file = csv.reader(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_ALL)
            for row in csv_file:
                feature = row
            file.close()
        return feature

    @staticmethod
    def __get_feature_command(feature, dataset):
        feature_command = str.split(feature, '_')
        feature_command = FeatureExtraction.__name_analyser(feature_command)
        group_search = re.search(r'x$|y$', feature_command[0])
        if group_search:
            axis = group_search.group(0)
            feature_command[0] = feature_command[0].replace(axis, '')
            axis += '_axis'
        else:
            axis = None
        command = {
            'MAIN_FEATURE': feature_command[0],
            'AXIS': axis,
            'OTHER_PARAMETERS': feature_command[1: len(feature_command)]
        }
        function = FeatureExtraction.__command_switch(feature_command)
        return function(command, dataset)

    @staticmethod
    def __name_analyser(split):
        switcher = {
            'STROKE': (FeatureExtraction.__join_command_name, 2),
            'TOTAL': (FeatureExtraction.__join_command_name, 2),
            'N': (FeatureExtraction.__join_command_name, 2),
            'NORM': (FeatureExtraction.__join_command_name, 2),
            'RATIO': (FeatureExtraction.__join_command_name, 2),
            'PEAK': (FeatureExtraction.__join_command_name, 2),
            'R': (FeatureExtraction.__join_command_name, 2),
            'SHANNON': (FeatureExtraction.__join_command_name, 2),
            'RENYI': (FeatureExtraction.__join_command_name, 2),
            'GL': (FeatureExtraction.__join_command_name, 2),
        }
        function, slices_number = switcher.get(split[0], (None, 0))
        if function is not None:
            name = [function(split, slices_number)]
            for i in range(slices_number, len(split)):
                name.append(split[i])
        else:
            name = split
        return name

    @staticmethod
    def __join_command_name(slices, slices_number):
        name = ""
        for i in range(slices_number):
            name += slices[i]
            if i < slices_number - 1:
                name += "_"
        return name

    @staticmethod
    def __command_switch(argument):
        switcher = {
            'DIS': FeatureExtraction.__get_displacement,
            'VEL': FeatureExtraction.__get_velocity,
            'ACC': FeatureExtraction.__get_acceleration,
            'JERK': FeatureExtraction.__get_jerk,
            'STROKE_SIZE': FeatureExtraction.__get_stroke_size,
            'STROKE_DURATION': FeatureExtraction.__get_stroke_duration,
            'STROKE_SPEED': FeatureExtraction.__get_stroke_speed,
            'TIME': FeatureExtraction.__get_time,
            'TOTAL_TIME': FeatureExtraction.__get_total_time,
            'N_STROKE': FeatureExtraction.__get_number_of_stroke,
            'NORM_TIME': FeatureExtraction.__get_normalized_time,
            'RATIO_TIME': FeatureExtraction.__get_ratio_time,
            'RATIO_PUPD': FeatureExtraction.__get_ration_pupd,
            'PEAK_VEL': FeatureExtraction.__get_peak_velocity,
            'PEAK_ACC': FeatureExtraction.__get_peak_velocity,
            'NCV': FeatureExtraction.__get_number_velocity_change,
            'NCA': FeatureExtraction.__get_number_acceleration_change,
            'R_NCV': FeatureExtraction.__get_ratio_number_velocity_change,
            'R_NCA': FeatureExtraction.__get_ratio_number_acceleration_change,
            'PRESSURE': FeatureExtraction.__get_pressure,
            'NCP': FeatureExtraction.__get_number_change_pressure,
            'NVV': FeatureExtraction.__get_normalized_velocity_variability,
            'SHANNON_ENTROPY': FeatureExtraction.__get_shannon_entropy,
            'RENYI_ENTROPY': FeatureExtraction.__get_renyi_entropy,
            'SNR': FeatureExtraction.__get_signal_to_noise_ratio,
            'DCT': FeatureExtraction.__get_discrete_cosine_transform,
            'DFT': FeatureExtraction.__get_discrete_fourier_transform,
            'RCEP': FeatureExtraction.__get_real_cepstrum,
            'GL_FD': FeatureExtraction.__get_fractional_derivative
        }
        function = switcher.get(argument[0], lambda: 'Invalid argument')
        return function

    @staticmethod
    def __get_finding_status(status):
        return ON_AIR if status == 'ia' else ON_SURFACE

    @staticmethod
    def __get_displacement(command, data):
        other_parameters = command['OTHER_PARAMETERS']
        axis = command['AXIS']
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
            if axis is None:
                displacement = Features.get_displacement_pen_status(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']],
                                                                    data[TASK_STRUCTURE['pen_status']], finding_status)
            else:
                displacement = Features.get_displacement_axis_pen_status(data[TASK_STRUCTURE[axis]], data[TASK_STRUCTURE['pen_status']],
                                                                         finding_status)
        else:
            if axis is None:
                displacement = Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']])
            else:
                displacement = Features.get_displacement_axis(data[TASK_STRUCTURE[axis]])
        return FeatureExtraction.__get_statistic_value(displacement)

    @staticmethod
    def __get_velocity(command, data):
        print(command)

    @staticmethod
    def __get_acceleration(command, data):
        print(command)

    @staticmethod
    def __get_jerk(command, data):
        print(command)

    @staticmethod
    def __get_stroke_size(command, data):
        print(command)

    @staticmethod
    def __get_stroke_duration(command, data):
        print(command)

    @staticmethod
    def __get_stroke_speed(command, data):
        print(command)

    @staticmethod
    def __get_time(command, data):
        print(command)

    @staticmethod
    def __get_total_time(command, data):
        print(command)

    @staticmethod
    def __get_number_of_stroke(command, data):
        print(command)

    @staticmethod
    def __get_normalized_time(command, data):
        print(command)

    @staticmethod
    def __get_ratio_time(command, data):
        print(command)

    @staticmethod
    def __get_ration_pupd(command, data):
        print(command)

    @staticmethod
    def __get_peak_velocity(command, data):
        print(command)

    @staticmethod
    def __get_peak_acceleration(command, data):
        print(command)

    @staticmethod
    def __get_number_velocity_change(command, data):
        print(command)

    @staticmethod
    def __get_number_acceleration_change(command, data):
        print(command)

    @staticmethod
    def __get_ratio_number_velocity_change(command, data):
        print(command)

    @staticmethod
    def __get_ratio_number_acceleration_change(command, data):
        print(command)

    @staticmethod
    def __get_pressure(command, data):
        print(command)

    @staticmethod
    def __get_number_change_pressure(command, data):
        print(command)

    @staticmethod
    def __get_normalized_velocity_variability(command, data):
        print(command)

    @staticmethod
    def __get_shannon_entropy(command, data):
        print(command)

    @staticmethod
    def __get_renyi_entropy(command, data):
        print(command)

    @staticmethod
    def __get_signal_to_noise_ratio(command, data):
        print(command)

    @staticmethod
    def __get_discrete_cosine_transform(command, data):
        print(command)

    @staticmethod
    def __get_discrete_fourier_transform(command, data):
        print(command)

    @staticmethod
    def __get_real_cepstrum(command, data):
        print(command)

    @staticmethod
    def __get_fractional_derivative(command, data):
        print(command)