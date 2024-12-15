import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracy(y_pred, y_true):
    return (y_pred == y_true).sum().item() / len(y_true)

def calculate_confusion_matrix(y_pred, y_true):
    # Calcola la confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Visualizza la confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def calculate_f1_score(y_pred, y_true):
    # Calcola la confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calcola i valori per la F1 score
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calcola la F1 score
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def calculate_precision(y_pred, y_true):
    # Calcola la confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calcola i valori per la precision
    FP = cm[0, 1]
    TP = cm[1, 1]

    # Calcola la precision
    precision = TP / (TP + FP)

    return precision

def calculate_recall(y_pred, y_true):
    # Calcola la confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calcola i valori per il recall
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calcola il recall
    recall = TP / (TP + FN)

    return recall
