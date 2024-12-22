import torch.nn as nn
from CNNTrainingTest import testCNN, trainingCNN
from MyLeNetCNN import MyLeNetCNN
from PrepareData import prepare_data
import torchvision
from PerformanceEvaluation import *
from StreamEvaluation import streamEvaluation

# Set number of experiments
num_exp = 10
image_path = 'D:\\Users\\Patrizio\\Desktop\\Hands'
num_train = 200
num_test = 100

#leNet = MyLeNetCNN(num_classes=2)
alexNet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)

# Update the final layer to output 2 classes
num_features = alexNet.classifier[6]. in_features
alexNet.classifier[6] = nn.Linear(num_features, 2)

# Freeze all layers except the newly added fully connected layer
for param in alexNet.parameters():
    param.requires_grad = False
for param in alexNet.classifier[6].parameters():
    param.requires_grad = True

# Prepare data
data_struct = prepare_data(num_exp=num_exp, num_train=num_train, num_test=num_test)

# Training the networks
train_loss_p = trainingCNN(net=alexNet, data_struct=data_struct, image_path=image_path, palmar_dorsal='palmar', tot_exp=num_exp)
train_loss_d = trainingCNN(net=alexNet, data_struct=data_struct, image_path=image_path, palmar_dorsal='dorsal', tot_exp=num_exp)

# Test the networks
ln_labels, ln_predicted = testCNN(net=alexNet, data_struct=data_struct, image_path=image_path, palmar_dorsal='palmar', tot_exp=num_exp)
an_labels, an_predicted = testCNN(net=alexNet, data_struct=data_struct, image_path=image_path, palmar_dorsal='dorsal', tot_exp=num_exp)

# Evaluate the unified network
print("Addestramento Reti Neurali Concluso")
un_labels, un_predicted  = streamEvaluation(net1=alexNet, net2=alexNet, data_struct=data_struct, image_path=image_path, tot_exp=num_exp)

# Performance evaluation
calculate_confusion_matrix(ln_labels, ln_predicted)
calculate_confusion_matrix(an_labels, an_predicted)
calculate_confusion_matrix(un_labels, un_predicted)

# Calculate the loss plot
calculate_loss_plot(train_loss_p)
calculate_loss_plot(train_loss_d)

# Calculate the accuracy plot
#calculate_accuracy_plot(ln_his_labels, ln_his_predicted)
#calculate_accuracy_plot(an_his_labels, an_his_predicted)
#calculate_accuracy_plot(un_his_label, un_his_predicted)

# Print the performance metrics
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