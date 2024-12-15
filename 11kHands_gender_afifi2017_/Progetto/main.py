import torch.nn as nn
from CNNTrainingTest import testCNN, trainingCNN
from MyLeNetCNN import MyLeNetCNN
from PrepareData import prepare_data
import torchvision
from PerformanceEvaluation import *
from StreamEvaluation import streamEvaluation

# Set number of experiments
num_exp = 5
image_path = '/home/mattpower/Downloads/Hands'
num_train = 50
num_test = 1

leNet = MyLeNetCNN(num_classes=2)
alexNet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)

# Update the final layer to output 2 classes
num_features = alexNet.classifier[6].in_features
alexNet.classifier[6] = nn.Linear(num_features, 2)

# Freeze all layers except the newly added fully connected layer
for param in alexNet.parameters():
    param.requires_grad = False
for param in alexNet.classifier[6].parameters():
    param.requires_grad = True

# Prepare data
data_struct = prepare_data(num_exp=num_exp, num_train=num_train, num_test=num_test)

# Training the networks
trainingCNN(net=leNet, data_struct=data_struct, image_path=image_path, palmar_dorsal='palmar', tot_exp=num_exp)
trainingCNN(net=alexNet, data_struct=data_struct, image_path=image_path, palmar_dorsal='dorsal', tot_exp=num_exp)

# Test the networks
ln_labels, ln_predicted = testCNN(net=leNet, data_struct=data_struct, image_path=image_path, palmar_dorsal='palmar', tot_exp=num_exp)
an_labels, an_predicted = testCNN(net=alexNet, data_struct=data_struct, image_path=image_path, palmar_dorsal='dorsal', tot_exp=num_exp)

# Evaluate the unified network
print("Addestramento Reti Neurali Concluso")
un_labels, un_predicted = streamEvaluation(net1=leNet, net2=alexNet, data_struct=data_struct, image_path=image_path, tot_exp=num_exp)

# Performance evaluation
calculate_confusion_matrix(ln_labels, ln_predicted)
calculate_confusion_matrix(an_labels, an_predicted)
calculate_confusion_matrix(un_labels, un_predicted)

print("\nAccuracy LeNet: ", calculate_accuracy(ln_labels, ln_predicted))
print("Precision LeNet: ", calculate_precision(ln_labels, ln_predicted))
print("Recall LeNet: ", calculate_recall(ln_labels, ln_predicted))
print("F1 Score LeNet: ", calculate_f1_score(ln_labels, ln_predicted),"\n")

print("\nAccuracy AlexNet: ", calculate_accuracy(an_labels, an_predicted))
print("Precision AlexNet: ", calculate_precision(an_labels, an_predicted))
print("Recall AlexNet: ", calculate_recall(an_labels, an_predicted))
print("F1 Score AlexNet: ", calculate_f1_score(an_labels, an_predicted),"\n")

print("\nAccuracy Unified Network: ", calculate_accuracy(un_labels, un_predicted))
print("Precision Unified Network: ", calculate_precision(un_labels, un_predicted))
print("Recall Unified Network: ", calculate_recall(un_labels, un_predicted))
print("F1 Score Unified Network: ", calculate_f1_score(un_labels, un_predicted),"\n")