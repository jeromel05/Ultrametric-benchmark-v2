import numpy as np
import torch
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import cm
import itertools

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
# one hot encoding for classes
def one_hot_labels(label, nb_classes):
    one_hot_vec = torch.zeros(nb_classes, dtype=torch.float)
    one_hot_vec[label]=torch.tensor(1, dtype=torch.float)
    return one_hot_vec

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    
    return image

def make_confusion_matrix_figure(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e3), decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def make_roc_curves_figure(fpr, tpr, classes):
    roc_curve_fig, a = plt.subplots(int(classes/4),int(classes/2))
    roc_curve_fig.set_figheight(10)
    roc_curve_fig.set_figwidth(10)
    for i in range(len(tpr)):
        a1 = a[i//4, i%4]
        a1.plot(fpr[i].cpu(), tpr[i].cpu())
        a1.set_xlabel("FPR")
        a1.set_ylabel("TPR")
        a1.set_title(f"ROC curve {i}")
    return roc_curve_fig

def print_metrics(acc, ap, auroc_, cf_mat, roc_curve, seed=0, do_print=False, save_figs=False):
    print()
    if do_print:
        print("----------------------------------------")
        print(f"""{bcolors.OKCYAN}Final validation accuracy: {acc:.3}
        Final validation ap: {ap:.3}
        Final validation auroc_: {auroc_:.3} {bcolors.ENDC}""")
    if save_figs:
        cf_mat.savefig(f"../../plots/best_val_cf_mat_seed_{seed}.jpg", dpi=250)
        roc_curve.savefig(f"../../plots/best_val_roc_curve_seed_{seed}.jpg", dpi=250)