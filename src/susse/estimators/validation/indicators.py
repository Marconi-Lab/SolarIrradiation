import numpy as np

class StatisticalIndicators:
    def __init__(self, observations, predictions):
        self.observations = np.array(observations)
        self.predictions = np.array(predictions)
        self.deviations = predictions - observations

    def mean_bias_deviation(self):
        return np.mean(self.deviations)

    def mean_absolute_deviation(self):
        return np.mean(np.abs(self.deviations))

    def root_mean_square_deviation(self):
        return np.sqrt(np.mean(np.square(self.deviations)))

    def relative_root_mean_square_deviation(self):
        return (self.root_mean_square_deviation() / np.mean(self.observations)) * 100