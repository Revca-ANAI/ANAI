import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from colorama import Fore


def confusion_matrix_plot(y_pred, y_val):
    """
    Takes Predicted data and Validation data as input and prepares and plots Confusion Matrix.
    """
    try:
        cm = confusion_matrix(y_val, y_pred)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt="g", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        ax.xaxis.set_ticklabels(np.unique(y_val))
        ax.yaxis.set_ticklabels(np.unique(y_val))
        plt.show()
    except Exception as error:
        print(Fore.RED + "Building Confusion Matrix Failed with error :", error, "\n")
