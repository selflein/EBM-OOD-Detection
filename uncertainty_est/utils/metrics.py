import torch
import numpy as np


def accuracy(y: torch.Tensor, y_hat: torch.Tensor):
    return (y == y_hat.argmax(dim=1)).float().mean(0).item()


def brier_score(labels, probs):
    probs = probs.clone()
    probs[np.arange(len(probs)), labels] -= 1
    score = (probs ** 2).sum(1).mean(0)
    return score


def classification_calibration(labels, probs, bins=10, tag=""):
    preds = np.argmax(probs, axis=1)
    total = labels.shape[0]
    probs = np.max(probs, axis=1)
    lower = 0.0
    increment = 1.0 / bins
    upper = increment
    accs = np.zeros([bins + 1], dtype=np.float32)
    gaps = np.zeros([bins + 1], dtype=np.float32)
    ece = 0.0
    for i in range(bins):
        ind1 = probs >= lower
        ind2 = probs < upper
        ind = np.where(np.logical_and(ind1, ind2))[0]
        if len(ind) > 0:
            lprobs = probs[ind]
            lpreds = preds[ind]
            llabels = labels[ind]
            acc = np.mean(np.asarray(llabels == lpreds, dtype=np.float32))
            prob = np.mean(lprobs)
        else:
            acc = 0.0
            prob = 0.0
        ece += np.abs(acc - prob) * float(lprobs.shape[0])
        gaps[i] = np.abs(acc - prob)
        accs[i] = acc
        upper += increment
        lower += increment
    ece /= np.float(total)
    mce = np.max(np.abs(gaps))

    accs[-1] = 1.0
    return ece, mce
