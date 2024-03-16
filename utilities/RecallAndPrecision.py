import numpy as np
import torch


def TruePositive(actualv, predictedv):
    ...

def FalseNegative(actualv, predictedv):
    ...

def FalsePositive(actualv, predictedv):
    ...

def TrueNegative(actualv, predictedv):
    ...

def RecallAndPercisoin(actualv, predictedv):
    actualv = actualv.numpy(force=True)
    predictedv = predictedv.numpy(force=True)
    TP = TruePositive(actualv, predictedv)
    FN = FalseNegative(actualv, predictedv)
    FP = FalsePositive(actualv, predictedv)
    FN = TrueNegative(actualv, predictedv)
    recall = TP / (TP + FN)
    percisoin = TP / (TP + FP)
    return recall, percisoin, (TP, FN, FP, FN)