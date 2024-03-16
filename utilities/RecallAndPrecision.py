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