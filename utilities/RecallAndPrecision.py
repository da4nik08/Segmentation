import numpy as np
import torch


def TruePositive(actualv, predictedv):
    return np.sum((np.round(predictedv) == 1) & (actualv == 1))

def FalseNegative(actualv, predictedv):
    return np.sum((np.round(predictedv) == 0) & (actualv == 1))

def FalsePositive(actualv, predictedv):
    return np.sum((np.round(predictedv) == 1) & (actualv == 0))

def TrueNegative(actualv, predictedv):
    return np.sum((np.round(predictedv) == 0) & (actualv == 0))

def RecallAndPrecision(actualv, predictedv):
    actualv = actualv.numpy(force=True)
    predictedv = predictedv.numpy(force=True)
    TP = TruePositive(actualv, predictedv)
    FN = FalseNegative(actualv, predictedv)
    FP = FalsePositive(actualv, predictedv)
    FN = TrueNegative(actualv, predictedv)
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    percisoin = TP / (TP + FP) if TP + FP != 0 else 0
    return recall, percisoin, (TP, FN, FP, FN)


class Metrics():
    def __init__(self):
        self.recall_per_batches = 0
        self.precision_per_batches = 0
        self.metrics_per_batches = np.array([0, 0, 0, 0])

    def batch_step(self, actualv, predictedv):
        recall, precision, metrics = RecallAndPrecision(actualv, predictedv)
        self.recall_per_batches += recall
        self.precision_per_batches += precision
        self.metrics_per_batches += np.array(metrics)

    def batch_average(self, num_batches):
        self.recall_per_batches /= num_batches
        self.precision_per_batches /= num_batches
        self.metrics_per_batches /= num_batches

    def get_metrics(self):
        return self.recall_per_batches, self.precision_per_batches, self.metrics_per_batches