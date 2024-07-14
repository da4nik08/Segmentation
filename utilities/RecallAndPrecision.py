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

def GetMetrics(actualv, predictedv):
    actualv = actualv.numpy(force=True)
    predictedv = predictedv.numpy(force=True)
    TP = TruePositive(actualv, predictedv)
    FN = FalseNegative(actualv, predictedv)
    FP = FalsePositive(actualv, predictedv)
    TN = TrueNegative(actualv, predictedv)
    return TP, FN, FP, TN


class Metrics():
    def __init__(self):
        self.metrics_per_batches = np.array([0, 0, 0, 0], dtype=np.float32)

    def batch_step(self, actualv, predictedv):
        metrics = GetMetrics(actualv, predictedv)
        self.metrics_per_batches += np.array(metrics)

    def get_metrics(self):
        TP, FN, FP, TN = self.metrics_per_batches
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        precisoin = TP / (TP + FP) if TP + FP != 0 else 0
        return recall, precisoin, self.metrics_per_batches

    def get_metrics_dice_iou(self):
        TP, FN, FP, TN = self.metrics_per_batches
        dice = (2 * TP) / (2 * TP + FP + FN) if TP + FP + FN != 0 else 0
        iou = TP / (TP + FP + FN) if TP + FP + FN != 0 else 0
        return dice, iou