import numpy as np
import itertools
from matplotlib import pyplot as plt


def history_show(history, accuracy_file = None, loss_file = None):
    """
    Args:
        history (dict): A dictionary containing the training history of a model.
        accuracy_file (str, optional): The file path to save the accuracy plot. Defaults to None.
        loss_file (str, optional): The file path to save the loss plot. Defaults to None.

    """
    plt.plot(history['accuracy'])
    plt.title('model %s' % history['type'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    if accuracy_file is not None:
        plt.savefig(accuracy_file)
    plt.show()
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    if loss_file is not None:
        plt.savefig(loss_file)
    plt.show()

def confusion_show(labels, y_pred, y_true, normalize = False, confusion_file = None):
    """
    Args:
        labels: A list of class labels.
        y_pred: An array or a list of predicted class labels.
        y_true: An array or a list of true class labels.
        normalize: A boolean indicating whether to normalize the confusion matrix. Default is False.
        confusion_file: A string specifying the file path to save the confusion matrix plot. Default is None.

    """
    classes = len(labels)

    cm = np.bincount(classes * y_true.astype(np.int32) + y_pred, 
                minlength = classes**2).reshape(classes, classes) 
    if normalize:
        cm = cm.astype(np.float64) / cm.max()
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    
    plt.xticks(range(classes), labels, rotation=45)
    plt.yticks(range(classes), labels)
    plt.ylim(classes - 0.5, -0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if confusion_file is not None:
        plt.savefig(confusion_file)
    plt.show()