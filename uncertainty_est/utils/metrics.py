import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy(y: torch.Tensor, y_hat: torch.Tensor):
    return (y == y_hat.argmax(dim=1)).float().mean(0).item()


def brier_score(labels, probs):
    probs = probs.copy()
    probs[np.arange(len(probs)), labels] -= 1
    score = (probs ** 2).sum(1).mean(0)
    return score


def brier_decomposition(labels, probs):
    """Compute the decompositon of the Brier score into its three components
    uncertainty (UNC), reliability (REL) and resolution (RES).

    Brier score is given by `BS = REL - RES + UNC`. The decomposition requires partioning
    into discrete events. Partioning into probability classes `M_k` is done for `p_k > p_i`
    for all `i!=k`. This induces a error when compared to the Brier score.

    For more information on the partioning see
    Murphy, A. H. (1973). A New Vector Partition of the Probability Score, Journal of Applied Meteorology and Climatology, 12(4)

    Args:
        labels: Numpy array of shape (num_preds,) containing the groundtruth
         class in range [0, n_classes - 1].
        probs: Numpy array of shape (num_preds, n_classes) containing predicted
         probabilities for the classes.

    Returns:
        (uncertainty, resolution, relability): Additive components of the Brier
         score decomposition.
    """
    preds = np.argmax(probs, axis=1)
    conf_mat = confusion_matrix(labels, preds, labels=np.arange(probs.shape[1]))

    pbar = np.sum(conf_mat, axis=0)
    pbar = pbar / pbar.sum()

    dist_weights = np.sum(conf_mat, axis=1)
    dist_weights = dist_weights / dist_weights.sum()

    dist_mean = conf_mat / (np.sum(conf_mat, axis=1)[:, None] + 1e-7)

    uncertainty = np.sum(pbar * (1 - pbar))

    resolution = (pbar[:, None] - dist_mean) ** 2
    resolution = np.sum(dist_weights * np.sum(resolution, axis=1))

    prob_true = np.take(dist_mean, preds, axis=0)
    reliability = np.sum((prob_true - probs) ** 2, axis=1)
    reliability = np.mean(reliability)

    return uncertainty, resolution, reliability


def calc_bins(labels, probs, num_bins=10):
    bins = np.linspace(0.1, 1, num_bins)
    confs = np.max(probs, axis=1)
    binned = np.digitize(confs, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    for bin_idx in range(num_bins):
        in_bin = binned == bin_idx
        bin_probs = probs[in_bin]
        if len(bin_probs) > 0:
            bin_sizes[bin_idx] = len(bin_probs)
            bin_confs[bin_idx] = np.mean(confs[in_bin])
            bin_accs[bin_idx] = (np.argmax(bin_probs, axis=1) == labels[in_bin]).mean()
    return bins, binned, bin_accs, bin_confs, bin_sizes


def classification_calibration(labels, probs, num_bins=10):
    _, _, bin_accs, bin_confs, bin_sizes = calc_bins(labels, probs, num_bins)

    cal_errors = np.abs(bin_accs - bin_confs)
    mce = np.max(cal_errors)
    ece = np.sum(cal_errors * (bin_sizes / np.sum(bin_sizes)))

    return ece, mce
