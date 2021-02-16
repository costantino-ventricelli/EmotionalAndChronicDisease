# coding=utf-8

import numpy as np
from scipy.fft import dct
from scipy.fft import rfft
from scipy.fft import irfft
import differint.differint as df
from SupportVectorMachineClassifier.FeaturesManager import FeaturesManager


class FunctionFeature:

    @staticmethod
    def get_displacement(x_axis, y_axis):
        result = FunctionFeature.__get_result_array(len(x_axis), null_value=-1)
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
        other_status = FeaturesManager.get_other_status(finding_status)
        result = FunctionFeature.__get_result_array(len(x_axis), null_value=-1)
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
        result = FunctionFeature.__get_result_array(len(time_stamps), null_value=1)
        for i in range(len(time_stamps)):
            if i < len(time_stamps) - 1:
                displacement = time_stamps[i + 1] - time_stamps[i]
                result.append(displacement) if displacement != 0 else result.append(1)
            else:
                result.append(result[-1])
        return np.array(result)

    @staticmethod
    def get_time_stamp_displacement_pen_status(time_stamps, pen_status, finding_status):
        other_status = FeaturesManager.get_other_status(finding_status)
        result = FunctionFeature.__get_result_array(len(time_stamps), null_value=1)
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
    def get_displacement_velocity(displacements, time_stamps):
        result = []
        for i in range(len(displacements)):
            velocity = 0.00
            if time_stamps[i] > 0:
                velocity = displacements[i] / time_stamps[i]
            result.append(velocity)
        return np.array(result)

    @staticmethod
    def get_displacement_velocity(displacements, time_stamps, on_surface):
        result = []
        if on_surface:
            time_stamps[time_stamps == 0] = 1
        for i in range(len(displacements)):
            velocity = 0.00
            if time_stamps[i] > 0:
                velocity = displacements[i] / time_stamps[i]
            result.append(velocity)
        return np.array(result)

    @staticmethod
    def get_displacement_acceleration(velocities, time_stamps):
        result = []
        for i in range(len(velocities)):
            acceleration = 0.00
            if time_stamps[i] > 0:
                acceleration = velocities[i] / time_stamps[i]
            result.append(acceleration)
        return np.array(result)

    @staticmethod
    def get_displacement_acceleration(velocities, time_stamps, on_surface):
        results = []
        if on_surface:
            time_stamps[time_stamps == 0] = 1
        for i in range(len(velocities)):
            acceleration = 0.00
            if time_stamps[i] > 0:
                acceleration = velocities[i] / time_stamps[i]
            results.append(acceleration)
        return np.array(results)

    @staticmethod
    def get_jerk(accelerations, time_stamps):
        result = []
        for i in range(len(accelerations)):
            jerk = 0.00
            if time_stamps[i] > 0:
                jerk = accelerations[i] / time_stamps[i]
            result.append(jerk)
        return np.array(result)

    @staticmethod
    def get_jerk(accelerations, time_stamp, on_surface):
        result = []
        if on_surface:
            time_stamp[time_stamp == 0] = 1
        for i in range(len(accelerations)):
            jerk = 0.00
            if time_stamp[i] > 0:
                jerk = accelerations[i] / time_stamp[i]
            result.append(jerk)
        return np.array(result)

    @staticmethod
    def get_displacement_axis(axis):
        result = FunctionFeature.__get_result_array(len(axis), null_value=-1)
        for i in range(len(axis)):
            if i < len(axis):
                square_difference = np.square(axis[i + 1] - axis[i])
                result.append(np.sqrt(square_difference))
            else:
                result.append(result[-1])
        return np.array(result)

    @staticmethod
    def get_displacement_axis_pen_status(axis, pen_status, finding_status):
        result = FunctionFeature.__get_result_array(len(axis), null_value=-1)
        other_status = FeaturesManager.get_other_status(finding_status)
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
