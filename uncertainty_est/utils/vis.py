from uncertainty_est.utils.metrics import calc_bins, classification_calibration

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_reliability_graph(labels, probs, num_bins, ax=None):
    ece, mce = classification_calibration(labels, probs, num_bins)
    bins, _, bin_accs, _, _ = calc_bins(labels, probs, num_bins)

    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")

    ax.grid(color="gray", linestyle="dashed")

    ax.bar(bins, bins, width=0.1, alpha=0.3, edgecolor="black", color="r", hatch="\\")

    ax.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor="black", color="b")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=2)
    ax.set_aspect("equal", adjustable="box")

    ECE_patch = mpatches.Patch(color="green", label="ECE = {:.2f}%".format(ece * 100))
    MCE_patch = mpatches.Patch(color="red", label="MCE = {:.2f}%".format(mce * 100))
    ax.legend(handles=[ECE_patch, MCE_patch])

    return ax
