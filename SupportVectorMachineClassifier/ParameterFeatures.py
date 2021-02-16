# coding=utf-8

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
from scipy.signal import argrelmin, argrelmax
from PyEMD import EMD
from SupportVectorMachineClassifier.FeaturesManager import FeaturesManager
from SupportVectorMachineClassifier.FunctionFeatures import FunctionFeature
from DatasetManager.Costants import ON_SURFACE, ON_AIR


class ParameterFeature:

    @staticmethod
    def get_mean_stroke_size(x_axis, y_axis, pen_status, finding_status):
        result = 0
        if ParameterFeature.get_stroke(pen_status, finding_status) != 0:
            strokes = FunctionFeature.get_displacement_pen_status(x_axis, y_axis, pen_status, finding_status)
            total = np.sum(strokes)
            result = total / len(strokes)
        return result

    @staticmethod
    def get_mean_axis_stroke_size(axis, pen_status, finding_status):
        result = 0
        if ParameterFeature.get_stroke(pen_status, finding_status) != 0:
            strokes = FunctionFeature.get_displacement_axis_pen_status(axis, pen_status, finding_status)
            total = np.sum(strokes)
            result = total / len(strokes)
        return result

    @staticmethod
    def get_mean_stroke_duration(time_stamps, pen_status, finding_status):
        result = ParameterFeature.get_time(pen_status, time_stamps, finding_status) / ParameterFeature.get_stroke(pen_status, finding_status)
        return result

    @staticmethod
    def get_mean_stroke_velocity(x_axis, y_axis, pen_status, time_stamps, finding_status):
        result = 0.00
        if ParameterFeature.get_stroke(pen_status, finding_status) != 0:
            stroke = []
            displacement = 0.00
            total = 0.00
            for i in range(len(pen_status) - 1):
                if time_stamps[i] == finding_status:
                    square_difference_x = np.square(x_axis[i + 1] - x_axis[i])
                    square_difference_y = np.square(y_axis[i + 1] - y_axis[i])
                    displacement += np.sqrt(square_difference_x - square_difference_y)
                    stroke.append(time_stamps[i])
                else:
                    delta_time = stroke[-1] - stroke[0]
                    if delta_time > 0.00:
                        total += displacement / delta_time
                    displacement = 0.00
                    total = 0.00
            delta_time = stroke[-1] - stroke[0]
            if delta_time > 0.00:
                total += displacement / delta_time
            result = total / ParameterFeature.get_stroke(pen_status, finding_status)
        return result

    @staticmethod
    def get_time(pen_status, time_stamps, finding_status):
        total_time = 0.00
        if ParameterFeature.get_stroke(pen_status, finding_status) != 0:
            stroke = []
            for i in range(len(time_stamps)):
                if time_stamps[i] == finding_status:
                    stroke.append(time_stamps[i])
                else:
                    if len(stroke) > 0:
                        total_time += stroke[-1] - stroke[0]
                        stroke = []
            if len(stroke) > 0:
                total_time += stroke[-1] - stroke[0]
        return total_time

    @staticmethod
    def get_total_time(time_stamps):
        return time_stamps[-1] - time_stamps[0]

    @staticmethod
    def get_total_time_norm(time, total_time):
        return time / total_time

    @staticmethod
    def get_ratio_time(time_on_surface, time_on_air):
        return  time_on_surface/time_on_air

    @staticmethod
    def get_pen_status_ratio(pen_status):
        return ParameterFeature.get_stroke(pen_status, ON_SURFACE) / ParameterFeature.get_stroke(pen_status, ON_AIR)

    @staticmethod
    def get_mean_function_peak(pen_status, function):
        result = 0.00
        if ParameterFeature.get_stroke(pen_status, None) != 0:
            function_strokes = []
            total = 0.00
            for i in range(len(pen_status) - 1):
                function_strokes.append(function[i])
                if pen_status[i + 1] == ON_SURFACE and pen_status[i] == ON_AIR or \
                        pen_status[i + 1] == ON_AIR and pen_status[i] == ON_SURFACE:
                    total += max(function_strokes)
                    function_strokes = []
            total += max(function_strokes)
            result = total / ParameterFeature.get_stroke(pen_status, None)
        return result

    @staticmethod
    def get_changes(function, pen_status):
        result = 0.00
        if ParameterFeature.get_stroke(pen_status, None) != 0:
            function_strokes = []
            total = 0.00
            for i in range(len(pen_status) - 1):
                function_strokes.append(function[i])
                if pen_status[i + 1] == ON_SURFACE and pen_status[i] == ON_AIR or \
                        pen_status[i + 1] == ON_AIR and pen_status[i] == ON_SURFACE:
                    total += len(argrelmin(np.array(function_strokes))[0])
                    total += len(argrelmax(np.array(function_strokes))[0])
                    function_strokes = []
            total += len(argrelmin(np.array(function_strokes))[0])
            total += len(argrelmax(np.array(function_strokes))[0])
            result = total / ParameterFeature.get_stroke(pen_status, None)
        return result

    @staticmethod
    def get_relative_changes(changes, total_changes):
        return changes / total_changes

    @staticmethod
    def get_mean_pressure(pressure, pen_status):
        result = 0.00
        if ParameterFeature.get_stroke(pen_status, None) != 0:
            pressure_on_surface = []
            total = 0.00
            for i in range(len(pressure)):
                if pen_status[i] == ON_SURFACE:
                    pressure_on_surface.append(pressure[i])
                else:
                    total += np.mean(pressure_on_surface)
                    pressure_on_surface = []
            total += np.mean(pressure_on_surface)
            result = total / ParameterFeature.get_stroke(pen_status, ON_SURFACE)
        return result

    @staticmethod
    def get_pressure_changes(pressure, pen_status):
        result = 0.00
        if ParameterFeature.get_stroke(pen_status, None) != 0:
            pressure_on_surface = []
            total = 0.00
            for i in range(len(pressure)):
                if pressure[i] == ON_SURFACE:
                    pressure_on_surface.append(pressure[i])
                else:
                    total += len(argrelmin(np.array(pressure_on_surface))[0])
                    total += len(argrelmax(np.array(pressure_on_surface))[0])
                    pressure_on_surface = []
            total += len(argrelmax(np.array(pressure_on_surface))[0])
            total += len(argrelmax(np.array(pressure_on_surface))[0])
            result = total / ParameterFeature.get_stroke(pen_status, None)
        return result

    @staticmethod
    def get_normalized_velocity_variability(velocity, total_time):
        velocity_change = []
        result = 0.00
        if total_time != 0:
            total = 0.00
            for i in range(len(velocity) - 1):
                total += np.abs(velocity[i + 1] - velocity[i])
                velocity_change.append(velocity[i + 1] - velocity[i])
            result = total / (total_time * np.abs(np.mean(velocity_change)))
        return result

    @staticmethod
    def get_renyi_entropy(x_probability, alpha):
        sum = 0.00
        for i in range(len(x_probability)):
            sum += np.power(x_probability[i], alpha)
        return 1/(1-alpha) * np.log2(sum)

    @staticmethod
    def get_der_snr(flux):
        flux = np.array(flux[np.where(np.array(flux) != 0.0)])
        n = len(flux)
        result = 0.00
        if n > 4:
            signal = np.median(flux)
            noise = 0.6052697 * np.median(np.abs(2.0 * flux[2: n-2] - flux[0: n-4] - flux[4: n]))
            result = float(signal / noise)
        return result

    @staticmethod
    def get_point_on(point, pen_status, finding_status):
        result = []
        for i in range(len(point)):
            if pen_status[i] == finding_status:
                result.append(point[i])
        if len(result) == 0:
            result.append(1)
        return result

    @staticmethod
    def get_calculate_imf(point):
        point = list(map(float, point))
        return EMD().emd(np.array(point), max_imf=2)

    @staticmethod
    def get_shannon_entropy(point, finding_status, pen_status):
        if finding_status is not None:
            point = ParameterFeature.get_point_on(point, pen_status, finding_status)
        samples = np.array(point).reshape(-1, 1)
        kernel_density = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(samples)
        log_probability = kernel_density.score_sampes(samples)
        probability = np.exp(log_probability)
        return entropy(probability)

    @staticmethod
    def get_imf_shannon_entropy(imf):
        samples = np.array(imf[0]).reshape(-1, 1)
        kernel_density = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(samples)
        log_probability = kernel_density.score_samples(samples)
        probability = np.exp(log_probability)
        return entropy(probability)

    @staticmethod
    def get_renyi_entropy(point, finding_status, pen_status, order):
        if finding_status is not None:
            point = ParameterFeature.get_point_on(point, pen_status, finding_status)
        samples = np.array(point).reshape(-1, 1)
        kernel_density = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(samples)
        log_probability = kernel_density.score_samples(samples)
        probability = np.exp(log_probability)
        return ParameterFeature.get_renyi_entropy(probability, order)

    @staticmethod
    def get_imf_renyi_entropy(imf, order):
        samples = np.array(imf[0]).reshape(-1, 1)
        kernel_density = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(samples)
        log_probability = kernel_density.score_samples(samples)
        probability = np.exp(log_probability)
        return ParameterFeature.get_renyi_entropy(probability, order)

    @staticmethod
    def get_snr(point, finding_status, pen_status):
        if finding_status is not None:
            point = ParameterFeature.get_point_on(point, pen_status, finding_status)
        return ParameterFeature.get_der_snr(point)

    @staticmethod
    def get_imf_snr(imf):
        return ParameterFeature.get_der_snr(imf[0])

    @staticmethod
    def get_imf_2_snr(imf):
        if len(imf) == 1:
            result = ParameterFeature.get_der_snr(imf[0])
        else:
            result = ParameterFeature.get_der_snr(imf[1])
        return result

    @staticmethod
    def get_stroke(pen_status, finding_status):
        result = 0
        total = False
        if finding_status is None:
            total = True
            finding_status = 1
        other_status = FeaturesManager.get_other_status(finding_status)
        for i in range(len(pen_status) - 1):
            if pen_status[i] == finding_status and pen_status[i + 1] == other_status:
                result += 1
            if pen_status[i] == other_status and pen_status[i + 1] == finding_status and total:
                result += 1
        return result
