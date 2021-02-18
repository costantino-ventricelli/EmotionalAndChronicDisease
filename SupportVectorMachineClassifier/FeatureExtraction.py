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
RESOURCE_FOLDER = 'resource'

MAIN_FEATURE_KEY = 'MAIN_FEATURE'
PARAMETER_KEY = 'OTHER_PARAMETERS'
AXIS_KEY = 'AXIS'


class FeatureExtraction:

    @staticmethod
    def get_features_for_task(dataset):
        task_file = FeatureExtraction.__get_header_file()
        features = FeatureExtraction.__get_feature_list()
        for feature in features:
            result = FeatureExtraction.__get_feature_command(feature, dataset)
            for key, item in result.items():
                try:
                    task_file = FeatureExtraction.__update_dictionary(task_file, key, item)
                except KeyError as error:
                    print(error)
        return task_file

    @staticmethod
    def __update_dictionary(dictionary, key, item):
        if key in dictionary.keys():
            value = dictionary[key]
            if value is None:
                dictionary[key] = [item]
            else:
                value.append(item)
        else:
            raise KeyError("The key: " + key + " doesn't exist")
        return dictionary

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
        with open(os.path.join(RESOURCE_FOLDER, FEATURE_FILE_LIST), 'r') as file:
            csv_file = csv.reader(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_ALL)
            for row in csv_file:
                feature = row
            file.close()
        return feature

    @staticmethod
    def __get_header_file():
        with open(os.path.join(RESOURCE_FOLDER, HEADER_FILE), 'r') as file:
            csv_file = csv.reader(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_ALL)
            for row in csv_file:
                header_list = row
            file.close()
        header = dict.fromkeys(header_list, None)
        return header

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
            MAIN_FEATURE_KEY: feature_command[0],
            AXIS_KEY: axis,
            PARAMETER_KEY: feature_command[1: len(feature_command)]
        }
        function = FeatureExtraction.__command_switch(feature_command)
        return function(feature, command, dataset)

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
    def __get_displacement(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
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
        return FeatureExtraction.__generate_result_dictionary(displacement, feature)

    @staticmethod
    def __get_velocity(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
        if axis is None:
            displacement = Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']])
        else:
            displacement = Features.get_displacement_axis(data[TASK_STRUCTURE[axis]])
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
            velocity = Features.get_displacement_velocity(displacement, data[TASK_STRUCTURE['pen_status']], bool(finding_status))
        else:
            velocity = Features.get_displacement_velocity(displacement, data[TASK_STRUCTURE['pen_status']])
        return FeatureExtraction.__generate_result_dictionary(velocity, feature)

    @staticmethod
    def __get_acceleration(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
        if axis is None:
            displacement = Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']])
        else:
            displacement = Features.get_displacement_axis(data[TASK_STRUCTURE[axis]])
        velocity = Features.get_displacement_velocity(displacement, data[TASK_STRUCTURE['pen_status']])
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
            acceleration = Features.get_displacement_acceleration(velocity, data[TASK_STRUCTURE['timestamp']], bool(finding_status))
        else:
            acceleration = Features.get_displacement_acceleration(velocity, data[TASK_STRUCTURE['timestamp']])
        return FeatureExtraction.__generate_result_dictionary(acceleration, feature)

    @staticmethod
    def __get_jerk(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
        if axis is None:
            displacement = Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']])
        else:
            displacement = Features.get_displacement_axis(data[TASK_STRUCTURE[axis]])
        acceleration = Features.get_displacement_acceleration(Features.get_displacement_velocity(displacement, data[TASK_STRUCTURE['timestamp']]),
                                                              data[TASK_STRUCTURE['timestamp']])
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
            jerk = Features.get_jerk(acceleration, data[TASK_STRUCTURE['timestamp']], bool(finding_status))
        else:
            jerk = Features.get_jerk(acceleration, data[TASK_STRUCTURE['timestamp']])
        return FeatureExtraction.__generate_result_dictionary(jerk, feature)

    @staticmethod
    def __get_stroke_size(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
            if axis is None:
                strokes_size = Features.get_mean_stroke_size(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']],
                                                             data[TASK_STRUCTURE['pen_status']], finding_status)
            else:
                strokes_size = Features.get_mean_axis_stroke_size(data[TASK_STRUCTURE[axis]], data[TASK_STRUCTURE['pen_status']],
                                                                  finding_status)
        else:
            strokes_size = None
            print("Stroke size error")
        return {feature + "[mean]": [strokes_size]}

    @staticmethod
    def __get_stroke_duration(feature, command, data):
        finding_status = FeatureExtraction.__get_finding_status(command[PARAMETER_KEY][0])
        return {feature + "[mean]": Features.get_mean_stroke_duration(data[TASK_STRUCTURE['timestamp']], data[TASK_STRUCTURE['pen_status']],
                                                           finding_status)}

    @staticmethod
    def __get_stroke_speed(feature, command, data):
        finding_status = FeatureExtraction.__get_finding_status(command[PARAMETER_KEY][0])
        return {feature + "[mean]": Features.get_mean_stroke_velocity(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']],
                                                           data[TASK_STRUCTURE['pen_status']], data[TASK_STRUCTURE['timestamp']],
                                                           finding_status)}

    @staticmethod
    def __get_time(feature, command, data):
        finding_status = FeatureExtraction.__get_finding_status(command[PARAMETER_KEY][0])
        return {feature: Features.get_time(data[TASK_STRUCTURE['pen_status']], data[TASK_STRUCTURE['timestamp']],
                                           finding_status)}

    @staticmethod
    def __get_total_time(feature, command, data):
        return {feature: Features.get_total_time(data[TASK_STRUCTURE['timestamp']])}

    @staticmethod
    def __get_number_of_stroke(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
        else:
            finding_status = None
        return {feature: Features.get_stroke(data[TASK_STRUCTURE['pen_status']], finding_status)}

    @staticmethod
    def __get_normalized_time(feature, command, data):
        finding_status = FeatureExtraction.__get_finding_status(command[PARAMETER_KEY][0])
        time = Features.get_time(data[TASK_STRUCTURE['pen_status']], data[TASK_STRUCTURE['timestamp']], finding_status)
        total_time = Features.get_total_time(data[TASK_STRUCTURE['timestamp']])
        return {feature: Features.get_total_time_norm(time, total_time)}

    @staticmethod
    def __get_ratio_time(feature, command, data):
        time_on_surface = Features.get_time(data[TASK_STRUCTURE['pen_status']], data[TASK_STRUCTURE['timestamp']], ON_SURFACE)
        time_in_air = Features.get_time(data[TASK_STRUCTURE['pen_status']], data[TASK_STRUCTURE['timestamp']], ON_AIR)
        return {feature: Features.get_ratio_time(time_on_surface, time_in_air)}

    @staticmethod
    def __get_ration_pupd(feature, command, data):
        return {feature: Features.get_pen_status_ratio(data[TASK_STRUCTURE['pen_status']])}

    @staticmethod
    def __get_peak_velocity(feature, command, data):
        velocity = Features.get_displacement_velocity(Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']]),
                                                      data[TASK_STRUCTURE['timestamp']])
        return {feature + "[mean]": Features.get_mean_function_peak(data[TASK_STRUCTURE['pen_status']], velocity)}

    @staticmethod
    def __get_peak_acceleration(feature, command, data):
        acceleration = Features.get_displacement_acceleration(Features.get_displacement_velocity(Features.get_displacement(data[TASK_STRUCTURE['x_axis']],
                                                            data[TASK_STRUCTURE['y_axis']]), data[TASK_STRUCTURE['timestamp']]), data[TASK_STRUCTURE['timestamp']])
        return {feature + "[mean]": Features.get_mean_function_peak(data[TASK_STRUCTURE['pen_status']], acceleration)}

    @staticmethod
    def __get_number_velocity_change(feature, command, data):
        velocity = Features.get_displacement_velocity(Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']]),
                                                      data[TASK_STRUCTURE['timestamp']])
        return {feature: Features.get_changes(velocity, data[TASK_STRUCTURE['pen_status']])}

    @staticmethod
    def __get_number_acceleration_change(feature, command, data):
        acceleration = Features.get_displacement_acceleration(Features.get_displacement_velocity(Features.get_displacement(data[TASK_STRUCTURE['x_axis']],
                                                              data[TASK_STRUCTURE['y_axis']]), data[TASK_STRUCTURE['timestamp']]), data[TASK_STRUCTURE['timestamp']])
        return {feature: Features.get_changes(acceleration, data[TASK_STRUCTURE['pen_status']])}

    @staticmethod
    def __get_ratio_number_velocity_change(feature, command, data):
        velocity = FeatureExtraction.__get_number_velocity_change(feature, None, data)
        time = FeatureExtraction.__get_total_time(feature, None, data)
        return {feature: Features.get_relative_changes(velocity[feature], time[feature])}

    @staticmethod
    def __get_ratio_number_acceleration_change(feature, command, data):
        acceleration = FeatureExtraction.__get_number_acceleration_change(feature, None, data)
        time = FeatureExtraction.__get_total_time(feature, None, data)
        return {feature: Features.get_relative_changes(acceleration[feature], time[feature])}

    @staticmethod
    def __get_pressure(feature, command, data):
        return {feature + "[mean]": Features.get_mean_pressure(data[TASK_STRUCTURE['pressure']], data[TASK_STRUCTURE['pen_status']])}

    @staticmethod
    def __get_number_change_pressure(feature, command, data):
        return {feature: Features.get_pressure_changes(data[TASK_STRUCTURE['pressure']], data[TASK_STRUCTURE['pen_status']])}

    @staticmethod
    def __get_normalized_velocity_variability(feature, command, data):
        velocity = Features.get_displacement_velocity(Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']]),
                                                      data[TASK_STRUCTURE['timestamp']])
        time = Features.get_total_time(data[TASK_STRUCTURE['timestamp']])
        return {feature: Features.get_normalized_velocity_variability(velocity, time)}

    @staticmethod
    def __get_shannon_entropy(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
        if len(other_parameters) > 0:
            if other_parameters[0] == 'os' or other_parameters[0] == 'ia':
                finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
                shannon_entropy = Features.get_shannon_entropy(data[TASK_STRUCTURE[axis]], finding_status, data[TASK_STRUCTURE['pen_status']])
            elif other_parameters[0] == '1Imf' or other_parameters[0] == '2Imf':
                shannon_entropy = Features.get_imf_shannon_entropy(Features.get_imf(data[TASK_STRUCTURE[axis]]))
            else:
                shannon_entropy = None
                print("Error in Shannon Entropy")
        else:
            shannon_entropy = Features.get_shannon_entropy(data[TASK_STRUCTURE[axis]])
        return {feature: shannon_entropy}

    @staticmethod
    def __get_renyi_entropy(feature, command, data):
        axis = command[AXIS_KEY]
        if len(command[PARAMETER_KEY]) > 1:
            status = command[PARAMETER_KEY][0]
            order = int(command[PARAMETER_KEY][1].replace('o', ''))
        else:
            status = None
            order = int(command[PARAMETER_KEY][0].replace('o', ''))
        if status is not None and status == 'ia' or status == 'os':
            finding_status = FeatureExtraction.__get_finding_status(status)
            renyi_entropy = Features.get_renyi_entropy(data[TASK_STRUCTURE[axis]], order, finding_status,
                                                       data[TASK_STRUCTURE['pen_status']])
        elif status is not None and status == '1Imf' or status == '1Imf':
            renyi_entropy = Features.get_imf_renyi_entropy(Features.get_imf(data[TASK_STRUCTURE[axis]]), order)
        else:
            renyi_entropy = Features.get_renyi_entropy(data[TASK_STRUCTURE[axis]], order)
        return {feature: renyi_entropy}

    @staticmethod
    def __get_signal_to_noise_ratio(feature, command, data):
        axis = command[AXIS_KEY]
        if len(command[PARAMETER_KEY]) > 0:
            if command[PARAMETER_KEY][0] == 'ia' or command[PARAMETER_KEY][0] == 'os':
                finding_status = FeatureExtraction.__get_finding_status(command[PARAMETER_KEY][0])
                snr = Features.get_snr(data[TASK_STRUCTURE[axis]], finding_status, data[TASK_STRUCTURE['pen_status']])
            else:
                snr = Features.get_imf_snr(Features.get_imf(data[TASK_STRUCTURE[axis]]))
        else:
            snr = Features.get_snr(data[TASK_STRUCTURE[axis]])
        return {feature: snr}

    @staticmethod
    def __get_discrete_cosine_transform(feature, command, data):
        other_parameter = command[PARAMETER_KEY]
        group_search = re.search(r'x$|y$', other_parameter[0])
        if group_search:
            axis = group_search.group(0)
            other_parameter[0] = other_parameter[0].replace(axis, '')
            point = {'axis': data[TASK_STRUCTURE[axis]]}
        else:
            point = {'x_axis': data[TASK_STRUCTURE['x_axis']],
                     'y_axis': data[TASK_STRUCTURE['y_axis']]}
        (function, parameter) = FeatureExtraction.__function_switch(other_parameter[0], data, other_parameter)
        # FIXME: quando vengono chiamati accelerzione, velocità e jerk se è specificato in aria o in superfice bisogna
        # selezionare solo il time stamp corrispondente.
        if len(other_parameter) > 1:
            finding_status = FeatureExtraction.__get_finding_status(other_parameter[1])
            function(point, parameter, finding_status)
        else:
            function(point, parameter)
        return FeatureExtraction.__generate_result_dictionary(Features.get_discrete_cosine_transform(function_result), feature)

    @staticmethod
    def __get_discrete_fourier_transform(feature, command, data):
        print(command)

    @staticmethod
    def __get_real_cepstrum(feature, command, data):
        print(command)

    @staticmethod
    def __get_fractional_derivative(feature, command, data):
        print(command)

    @staticmethod
    def __function_switch(function_name, data):
        switcher = {
            'DIS': (FeatureExtraction.__get_function_displacement, data[TASK_STRUCTURE['pen_status']]),
            'VEL': (FeatureExtraction.__get_function_velocity, data[TASK_STRUCTURE['timestamp']]),
            'ACC': (FeatureExtraction.__get_function_acceleration, data[TASK_STRUCTURE['timestamp']]),
            'JERK': (FeatureExtraction.__get_function_jerk, data[TASK_STRUCTURE['timestamp']]),
        }
        return switcher.get(function_name)

    @staticmethod
    def __get_function_displacement(point, pen_status, finding_status=None):
        if len(point.keys) > 1:
            if finding_status is not None:
                function = Features.get_displacement_pen_status(point['x_axis'], point['y_axis'], pen_status, finding_status)
            else:
                function = Features.get_displacement(point['x_axis'], point['y_axis'])
        else:
            if finding_status is not None:
                function = Features.get_displacement_axis_pen_status(point['axis'], pen_status, finding_status)
            else:
                function = Features.get_displacement_axis(point['axis'])
        return function

    @staticmethod
    def __get_function_velocity(point, time_stamp, on_surface=None):
        space = FeatureExtraction.__get_function_displacement(point)
        return Features.get_displacement_velocity(space, time_stamp, bool(on_surface))

    @staticmethod
    def __get_function_acceleration(point, time_stamp, on_surface=None):
        velocity = FeatureExtraction.__get_function_velocity(point, time_stamp, bool(on_surface))
        return Features.get_displacement_acceleration(velocity, time_stamp, on_surface)

    @staticmethod
    def __get_function_jerk(point, time_stamp, on_surface=None):
        acceleration = FeatureExtraction.__get_function_acceleration(point, time_stamp, bool(on_surface))
        return Features.get_jerk(acceleration, time_stamp, on_surface)

    @staticmethod
    def __generate_result_dictionary(data, feature):
        result = {}
        for key, item in FeatureExtraction.__get_statistic_value(data).items():
            result[feature + key] = item
        return result
