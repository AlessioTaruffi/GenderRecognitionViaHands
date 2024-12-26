from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracy(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    return (cm[1,1] + cm[0,0])/len(y_true)

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

def calculate_loss_plot(train_loss):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'bo-', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


'''
Not working :(
def calculate_accuracy_plot(tot_labels, predicted_labels):
    test_accuracy = []
    print(tot_labels)
    print("\n")
    print(predicted_labels)
    for i in range(len(tot_labels)):
       # test_accuracy.append((tot_labels[0:i+1] == predicted_labels[0:i+1]).sum().item() / len(tot_labels[0:i+1]))
        cm = confusion_matrix(tot_labels[i], tot_labels[i])
        test_accuracy.append(cm[1,1] /len(tot_labels[i]))
    epochs = range(1, len(test_accuracy) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_accuracy, 'ro-', label='Test accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
'''
