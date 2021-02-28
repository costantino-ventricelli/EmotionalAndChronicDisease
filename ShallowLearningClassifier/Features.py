# coding=utf-8

import numpy as np
import differint.differint as df

from scipy.fft import dct
from scipy.fft import rfft
from scipy.fft import irfft
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
from scipy.signal import argrelmin, argrelmax
from PyEMD import EMD
from DatasetManager.Costants import ON_SURFACE, ON_AIR
from copy import deepcopy


class Features:

    @staticmethod
    def get_point(point, pen_status, finding_status):
        new_point = deepcopy(point)
        for i in range(len(pen_status)):
            if pen_status[i] != finding_status:
                for key, item in point.items():
                    del new_point[key][new_point[key].index(item[i])]
        return new_point

    @staticmethod
    def get_displacement(x_axis, y_axis):
        result = Features.__get_result_array(len(x_axis), null_value=-1)
        for i in range(len(x_axis)):
            if i < len(x_axis) - 1:
                square_difference_x = np.square(x_axis[i + 1] - x_axis[i])
                square_difference_y = np.square(y_axis[i + 1] - y_axis[i])
                result.append(np.sqrt(square_difference_x + square_difference_y))
            else:
                result.append(result[-1])
        return np.array(result)

    @staticmethod
    def get_displacement_pen_status(x_axis, y_axis, pen_status, finding_status):
        other_status = Features.get_other_status(finding_status)
        result = Features.__get_result_array(len(x_axis), null_value=-1)
        for i in range(len(x_axis) - 1):
            if pen_status[i] == finding_status and pen_status[i + 1] == finding_status:
                square_difference_x = np.square(x_axis[i + 1] - x_axis[i])
                square_difference_y = np.square(y_axis[i + 1] - y_axis[i])
                result.append(np.sqrt(square_difference_x + square_difference_y))
            elif pen_status[i - 1] == finding_status and pen_status[i] == finding_status and pen_status[i + 1] == other_status:
                if len(result) == 0:
                    result.append(-1)
                result.append(result[-1])
            elif pen_status[i - 1] == other_status and pen_status[i] == finding_status and pen_status[i + 1] == other_status:
                result.append(0)
        return np.array(result)

    @staticmethod
    def get_time_stamp_displacement(time_stamps):
        result = Features.__get_result_array(len(time_stamps), null_value=1)
        for i in range(len(time_stamps)):
            if i < len(time_stamps) - 1:
                displacement = time_stamps[i + 1] - time_stamps[i]
                result.append(displacement) if displacement != 0 else result.append(1)
            else:
                result.append(result[-1])
        return np.array(result)

    @staticmethod
    def get_time_stamp_displacement_pen_status(time_stamps, pen_status, finding_status):
        other_status = Features.get_other_status(finding_status)
        result = Features.__get_result_array(len(time_stamps), null_value=1)
        for i in range(len(time_stamps) - 1):
            if pen_status[i] == finding_status and pen_status[i + 1] == finding_status:
                displacement = time_stamps[i + 1] - time_stamps[i]
                result.append(displacement) if displacement != 0 else result.append(1)
            elif pen_status[i - 1] == finding_status and pen_status[i] == finding_status and pen_status[i + 1] == other_status:
                result.append(-1) if len(result) == 0 else result.append(result[-1])
            elif pen_status[i - 1] == other_status and pen_status[i] == finding_status and pen_status[i + 1] == other_status:
                result.append(1)
        return np.array(result)

    @staticmethod
    def get_displacement_velocity(displacements, time_stamps, on_surface=None):
        result = []
        if on_surface is not None and on_surface:
            time_stamps[time_stamps == 0] = 1
        for i in range(len(displacements) - 1):
            velocity = 0.00
            if time_stamps[i] > 0.00:
                velocity = displacements[i] / time_stamps[i]
            result.append(velocity)
        return np.array(result)

    @staticmethod
    def get_displacement_acceleration(velocities, time_stamps, on_surface=None):
        results = []
        if on_surface is not None and on_surface:
            time_stamps[time_stamps == 0] = 1
        for i in range(len(velocities) - 1):
            acceleration = 0.00
            if time_stamps[i] > 0.00:
                acceleration = velocities[i] / time_stamps[i]
            results.append(acceleration)
        return np.array(results)

    @staticmethod
    def get_jerk(accelerations, time_stamp, on_surface=None):
        result = []
        if on_surface is not None and on_surface:
            time_stamp[time_stamp == 0] = 1
        for i in range(len(accelerations)):
            jerk = 0.00
            if time_stamp[i] > 0:
                jerk = accelerations[i] / time_stamp[i]
            result.append(jerk)
        return np.array(result)

    @staticmethod
    def get_displacement_axis(axis):
        result = Features.__get_result_array(len(axis), null_value=-1)
        for i in range(len(axis)):
            if i < len(axis) - 1:
                square_difference = np.square(axis[i + 1] - axis[i])
                result.append(np.sqrt(square_difference))
            else:
                result.append(result[-1])
        return np.array(result)

    @staticmethod
    def get_displacement_axis_pen_status(axis, pen_status, finding_status):
        result = Features.__get_result_array(len(axis), null_value=-1)
        other_status = Features.get_other_status(finding_status)
        for i in range(len(axis) - 1):
            if pen_status[i] == finding_status and pen_status[i + 1] == finding_status:
                square_difference = np.square(axis[i + 1] - axis[i - 1])
                result.append(np.sqrt(square_difference))
            elif pen_status[i - 1] == finding_status and pen_status[i] == finding_status and pen_status[i + 1] == other_status:
                if len(result) == 0:
                    result.append(-1)
                result.append(result[-1])
            elif pen_status[i - 1] == other_status and pen_status[i] == finding_status and pen_status[i + 1] == other_status:
                result.append(0)
        return np.array(result)

    @staticmethod
    def get_discrete_cosine_transform(function):
        return dct(function)

    @staticmethod
    def get_real_fast_fourier_transform(function):
        return rfft(function)

    @staticmethod
    def get_cepstrum(function):
        spectrum = rfft(function)
        return irfft(np.abs(spectrum))

    @staticmethod
    def get_fractional_derivative(function, alpha):
        if len(function) == 1:
            return [-1]
        else:
            return df.GLI(alpha, function, num_points=len(function))

    @staticmethod
    def __get_result_array(length, null_value):
        if length > 0:
            array = []
        else:
            array = [null_value]
        return array

    @staticmethod
    def get_mean_stroke_size(x_axis, y_axis, pen_status, finding_status):
        result = 0
        if Features.get_stroke(pen_status, finding_status) != 0:
            strokes = Features.get_displacement_pen_status(x_axis, y_axis, pen_status, finding_status)
            total = np.sum(strokes)
            if len(strokes) != 0:
                result = total / len(strokes)
        return result

    @staticmethod
    def get_mean_axis_stroke_size(axis, pen_status, finding_status):
        result = 0
        if Features.get_stroke(pen_status, finding_status) != 0:
            strokes = Features.get_displacement_axis_pen_status(axis, pen_status, finding_status)
            total = np.sum(strokes)
            if len(strokes) != 0:
                result = total / len(strokes)
        return result

    @staticmethod
    def get_mean_stroke_duration(time_stamps, pen_status, finding_status):
        result = 0.00
        stroke_number = Features.get_stroke(pen_status, finding_status)
        if stroke_number != 0:
            result = Features.get_time(pen_status, time_stamps, finding_status) / stroke_number
        return result

    @staticmethod
    def get_mean_stroke_velocity(x_axis, y_axis, pen_status, time_stamps, finding_status):
        result = 0.00
        if Features.get_stroke(pen_status, finding_status) != 0:
            stroke = []
            displacement = 0.00
            total = 0.00
            for i in range(len(pen_status) - 1):
                if pen_status[i] == finding_status:
                    square_difference_x = np.square(x_axis[i + 1] - x_axis[i])
                    square_difference_y = np.square(y_axis[i + 1] - y_axis[i])
                    displacement += np.sqrt(np.abs(square_difference_x - square_difference_y))
                    stroke.append(time_stamps[i])
                else:
                    if stroke:
                        delta_time = stroke[-1] - stroke[0]
                    else:
                        delta_time = 0.00
                    if delta_time > 0.00:
                        total += displacement / delta_time
                    displacement = 0.00
                    total = 0.00
            if stroke:
                delta_time = stroke[-1] - stroke[0]
            else:
                delta_time = 0.00
            if delta_time > 0.00:
                total += displacement / delta_time
            result = total / Features.get_stroke(pen_status, finding_status)
        return result

    @staticmethod
    def get_time(pen_status, time_stamps, finding_status):
        total_time = 0.00
        if Features.get_stroke(pen_status, finding_status) != 0:
            stroke = []
            for i in range(len(time_stamps)):
                if pen_status[i] == finding_status:
                    stroke.append(time_stamps[i])
                else:
                    if len(stroke) > 0:
                        total_time += stroke[-1] - stroke[0]
                        stroke = []
            if len(stroke) > 0:
                total_time += stroke[-1] - stroke[0]
        return total_time

    @staticmethod
    def get_time_function(time_stamp, pen_status, finding_status):
        result = []
        for i in range(len(time_stamp)):
            if pen_status[i] == finding_status:
                result.append(time_stamp[i])
        return result

    @staticmethod
    def get_total_time(time_stamps):
        return time_stamps[-1] - time_stamps[0]

    @staticmethod
    def get_total_time_norm(time, total_time):
        result = 0.00
        if total_time != 0:
            result = time / total_time
        return result

    @staticmethod
    def get_ratio_time(time_on_surface, time_on_air):
        result = 0.00
        if time_on_air != 0:
            result = time_on_surface / time_on_air
        return result

    @staticmethod
    def get_pen_status_ratio(pen_status):
        result = 0.00
        stroke_on_surface = Features.get_stroke(pen_status, ON_SURFACE)
        stroke_in_air = Features.get_stroke(pen_status, ON_AIR)
        if stroke_in_air != 0:
            result = stroke_on_surface / stroke_in_air
        return result

    @staticmethod
    def get_mean_function_peak(pen_status, function):
        result = 0.00
        if Features.get_stroke(pen_status, None) != 0:
            function_strokes = []
            total = 0.00
            for i in range(len(pen_status) - 1):
                function_strokes.append(function[i])
                if pen_status[i + 1] == ON_SURFACE and pen_status[i] == ON_AIR or \
                        pen_status[i + 1] == ON_AIR and pen_status[i] == ON_SURFACE:
                    if function_strokes:
                        total += max(function_strokes)
                    function_strokes = []
            if function_strokes:
                total += max(function_strokes)
            result = total / Features.get_stroke(pen_status, None)
        return result

    @staticmethod
    def get_changes(function, pen_status):
        result = 0.00
        if Features.get_stroke(pen_status, None) != 0:
            function_strokes = []
            total = 0.00
            for i in range(len(pen_status) - 1):
                function_strokes.append(function[i])
                if pen_status[i + 1] == ON_SURFACE and pen_status[i] == ON_AIR or \
                        pen_status[i + 1] == ON_AIR and pen_status[i] == ON_SURFACE:
                    if function_strokes:
                        total += len(argrelmin(np.array(function_strokes))[0])
                        total += len(argrelmax(np.array(function_strokes))[0])
                    function_strokes = []
            if function_strokes:
                total += len(argrelmin(np.array(function_strokes))[0])
                total += len(argrelmax(np.array(function_strokes))[0])
            result = total / Features.get_stroke(pen_status, None)
        return result

    @staticmethod
    def get_relative_changes(changes, total_time):
        result = 0.00
        if total_time != 0:
            result = changes / total_time
        return result

    @staticmethod
    def get_mean_pressure(pressure, pen_status):
        result = 0.00
        if Features.get_stroke(pen_status, None) != 0:
            pressure_on_surface = []
            total = 0.00
            for i in range(len(pressure)):
                if pen_status[i] == ON_SURFACE:
                    pressure_on_surface.append(pressure[i])
                else:
                    if pressure_on_surface:
                        total += np.mean(pressure_on_surface)
                    pressure_on_surface = []
            if pressure_on_surface:
                total += np.mean(pressure_on_surface)
            result = total / Features.get_stroke(pen_status, ON_SURFACE)
        return result

    @staticmethod
    def get_pressure_changes(pressure, pen_status):
        result = 0.00
        if Features.get_stroke(pen_status, None) != 0:
            pressure_on_surface = []
            total = 0.00
            for i in range(len(pressure)):
                if pressure[i] == ON_SURFACE:
                    pressure_on_surface.append(pressure[i])
                else:
                    if pressure_on_surface:
                        total += len(argrelmin(np.array(pressure_on_surface))[0])
                        total += len(argrelmax(np.array(pressure_on_surface))[0])
                    pressure_on_surface = []
            if pressure_on_surface:
                total += len(argrelmax(np.array(pressure_on_surface))[0])
                total += len(argrelmax(np.array(pressure_on_surface))[0])
            result = total / Features.get_stroke(pen_status, None)
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
    def __get_renyi_entropy(x_probability, alpha):
        sum = 0.00
        result = 0.00
        for i in range(len(x_probability)):
            sum += np.power(x_probability[i], alpha)
        if (1 - alpha) * np.log2(sum) != 0:
            result = 1 / (1 - alpha) * np.log2(sum)
        return result

    @staticmethod
    def get_der_snr(flux):
        flux = np.array(flux)
        flux = np.array(flux[np.where(flux != 0.0)])
        n = len(flux)
        result = 0.00
        if n > 4:
            signal = np.median(flux)
            noise = 0.6052697 * np.median(np.abs(2.0 * flux[2: n - 2] - flux[0: n - 4] - flux[4: n]))
            if noise != 0:
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
    def get_imf(point):
        point = list(map(float, point))
        return EMD().emd(np.array(point), max_imf=2)

    @staticmethod
    def get_shannon_entropy(point, finding_status=None, pen_status=None):
        if finding_status is not None:
            point = Features.get_point_on(point, pen_status, finding_status)
        samples = np.array(point).reshape((-1, 1))
        kernel_density = KernelDensity(kernel='gaussian', bandwidth=0.2)
        kernel_density.fit(samples)
        log_probability = kernel_density.score_samples(samples)
        probability = np.exp(log_probability)
        return entropy(probability)

    @staticmethod
    def get_imf_shannon_entropy(imf):
        samples = np.array(imf[0]).reshape(-1, 1)
        kernel_density = KernelDensity(kernel='gaussian', bandwidth=0.2)
        kernel_density.fit(samples)
        log_probability = kernel_density.score_samples(samples)
        probability = np.exp(log_probability)
        return entropy(probability)

    @staticmethod
    def get_renyi_entropy(point, order, finding_status=None, pen_status=None):
        if finding_status is not None:
            point = Features.get_point_on(point, pen_status, finding_status)
        samples = np.array(point).reshape(-1, 1)
        kernel_density = KernelDensity(kernel='gaussian', bandwidth=0.2)
        kernel_density.fit(samples)
        log_probability = kernel_density.score_samples(samples)
        probability = np.exp(log_probability)
        return Features.__get_renyi_entropy(probability, order)

    @staticmethod
    def get_imf_renyi_entropy(imf, order):
        samples = np.array(imf[0]).reshape(-1, 1)
        kernel_density = KernelDensity(kernel='gaussian', bandwidth=0.2)
        kernel_density.fit(samples)
        log_probability = kernel_density.score_samples(samples)
        probability = np.exp(log_probability)
        return Features.__get_renyi_entropy(probability, order)

    @staticmethod
    def get_snr(point, finding_status=None, pen_status=None):
        if finding_status is not None:
            point = Features.get_point_on(point, pen_status, finding_status)
        return Features.get_der_snr(point)

    @staticmethod
    def get_imf_snr(imf):
        return Features.get_der_snr(imf[0])

    @staticmethod
    def get_imf_2_snr(imf):
        if len(imf) == 1:
            result = Features.get_der_snr(imf[0])
        else:
            result = Features.get_der_snr(imf[1])
        return result

    @staticmethod
    def get_stroke(pen_status, finding_status=None):
        result = 0
        total = False
        if finding_status is None:
            total = True
            finding_status = 1
        other_status = Features.get_other_status(finding_status)
        for i in range(len(pen_status) - 1):
            if pen_status[i] == finding_status and pen_status[i + 1] == other_status:
                result += 1
            if pen_status[i] == other_status and pen_status[i + 1] == finding_status and total:
                result += 1
        return result
    
    @staticmethod
    def get_other_status(status):
        return 0 if status == 1 else 1
