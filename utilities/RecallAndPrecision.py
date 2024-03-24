import numpy as np
import torch


def TruePositive(actualv, predictedv):
    return np.sum(np.bool_(np.round(predictedv)) & np.bool_(actualv))

def FalseNegative(actualv, predictedv):
    return np.sum((np.round(predictedv) == 0) & np.bool_(actualv))

def FalsePositive(actualv, predictedv):
    return np.sum(np.bool_(np.round(predictedv)) & (actualv == 0))

def TrueNegative(actualv, predictedv):
    return np.sum((np.round(predictedv) == 0) & (actualv == 0))

def RecallAndPrecision(actualv, predictedv):
    actualv = actualv.numpy(force=True)
    predictedv = predictedv.numpy(force=True)
    TP = TruePositive(actualv, predictedv)
    FN = FalseNegative(actualv, predictedv)
    FP = FalsePositive(actualv, predictedv)
    TN = TrueNegative(actualv, predictedv)
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    precisoin = TP / (TP + FP) if TP + FP != 0 else 0
    return recall, precisoin, (TP, FN, FP, FN)


class Metrics():
    def __init__(self):
        self.recall_per_batches = 0
        self.precision_per_batches = 0
        self.metrics_per_batches = np.array([0, 0, 0, 0], dtype=np.float32)

    def batch_step(self, actualv, predictedv):
        recall, precision, metrics = RecallAndPrecision(actualv, predictedv)
        self.recall_per_batches += recall
        self.precision_per_batches += precision
        self.metrics_per_batches += np.array(metrics)

    def instance_average(self, num_instance):
        self.recall_per_batches /= num_instance
        self.precision_per_batches /= num_instance
        self.metrics_per_batches /= num_batches

    def get_metrics(self):
        return self.recall_per_batches, self.precision_per_batches, self.metrics_per_batches