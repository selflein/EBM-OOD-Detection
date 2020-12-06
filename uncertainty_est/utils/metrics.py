import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy(y: torch.Tensor, y_hat: torch.Tensor):
    return (y == y_hat.argmax(dim=1)).float().mean(0).item()


def brier_score(labels, probs):
    probs = probs.clone()
    probs[np.arange(len(probs)), labels] -= 1
    score = (probs ** 2).sum(1).mean(0)
    return score


def brier_decomposition(labels, probs):
    """Compute the decompositon of the Brier score into its three components
     uncertainty, reliability and resolution. Brier score is given by
     `BS = REL - RES + UNC`. Discretization into probability classes `M_k` is done for
     `p_k > p_i` for all `i!=k`. This induces a error when compared to the Brier score.

     Args:
        labels: Numpy array of shape (num_preds,) containing the groundtruth
         class in range [0, n_classes - 1].
        probs: Numpy array of shape (num_preds, n_classes) containing predicted
         probabilities for the classes.

    Returns:
        (uncertainty, resolution, relability): Additive components of the Brier
         score decomposition.
    """
    preds = np.argmax(probs, dim=1)
    conf_mat = confusion_matrix(labels, preds, labels=np.arange(probs.shape[1]))

    pbar = np.sum(conf_mat, axis=0)
    pbar /= pbar.sum()

    dist_weights = np.sum(conf_mat, axis=1)
    dist_weights /= dist_weights.sum()

    dist_mean = conf_mat / (np.sum(conf_mat, axis=1)[:, None] + 1e-7)

    uncertainty = -np.sum(pbar ** 2)

    resolution = (pbar[:, None] - dist_mean) ** 2
    resolution = np.sum(dist_weights * np.sum(resolution, axis=1))

    prob_true = np.take(dist_mean, preds, axis=0)
    reliability = np.sum((prob_true - probs) ** 2, axis=1)
    reliability = np.mean(reliability)

    return uncertainty, resolution, reliability


def classification_calibration(labels, probs, bins=10):
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
