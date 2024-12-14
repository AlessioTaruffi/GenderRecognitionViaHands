from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomTransform import CustomDorsalTransform, CustomPalmTransform
from DataLoader import CustomImageDataset
from CNNTrainingTest import trainingCNN

from MyLeNetCNN import MyLeNetCNN
from torch.utils.data import DataLoader
from CustomTransform import CustomDorsalTransform, CustomPalmTransform
from DataLoader import CustomImageDataset
from PrepareData import prepare_data
import torchvision


def streamEvaluation(net1:nn.Module, net2:nn.Module, data_struct:dict, image_path:str, tot_exp: int, batch_size=32):
    # Definisci le trasformazioni da applicare alle immagini (opzionale)
    palmar_transform = transforms.Compose([
        CustomPalmTransform(),
        transforms.ToTensor(),          # Converte le immagini in tensori
    ])

    dorsal_transform = transforms.Compose([
        CustomDorsalTransform(),
        transforms.ToTensor(),          # Converte le immagini in tensori
    ])

    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Carica la rete neurale
    net1.to(device)
    net2.to(device)
    
    net1.eval()
    net2.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for exp in range(tot_exp):
            dataset_dorsal = CustomImageDataset(image_dir=image_path, data_structure= data_struct, id_exp=exp, train_test='test', palmar_dorsal='dorsal', transform=[palmar_transform, dorsal_transform] )
            data_loader_dorsal = DataLoader(dataset_dorsal, batch_size=batch_size, shuffle=True)
           
            dataset_palmar = CustomImageDataset(image_dir=image_path, data_structure= data_struct, id_exp=exp, train_test='test', palmar_dorsal='palmar', transform=[palmar_transform, dorsal_transform] )
            data_loader_palmar= DataLoader(dataset_palmar, batch_size=batch_size, shuffle=True)

            for data_dorsal, data_palmar in zip(data_loader_dorsal, data_loader_palmar):
                
                dorsal_images, dorsal_labels = data_dorsal
                dorsal_images, dorsal_labels = dorsal_images.to(device), dorsal_labels.to(device)

                palmar_images, palmar_labels = data_palmar
                palmar_images, palmar_labels = palmar_images.to(device), palmar_labels.to(device)
                
                # Softmax layer
                outputs_leNet = net1(palmar_images)
                outputs_alexNet = net2(dorsal_images)

                # Score fusion layer
                #output_M = outputs_leNet[1] * 0.6  + outputs_alexNet[1] * 0.4
                #output_F = outputs_leNet[0] * 0.6  + outputs_alexNet[0] * 0.4

                # Applica la softmax agli output per ottenere le probabilità
                softmax = torch.nn.Softmax(dim=1)
                probs_leNet = softmax(outputs_leNet)
                probs_alexNet = softmax(outputs_alexNet)
    
                # Esegui la score fusion combinando le probabilità
                fused_probs = probs_leNet * 0.6 + probs_alexNet * 0.4
    
                # Ottieni la previsione finale
                _, predicted = torch.max(fused_probs, 1)

                #predicted = torch.cat((predicted1, predicted2), 1)
                total += palmar_labels.size(0)
                correct += (predicted == palmar_labels).sum().item()
    print(f'Accuracy on the test images: {100 * correct / total:.2f}%')


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

# Da mettere prima di richiamare le classi (reti)
data_struct = prepare_data(num_exp=10, num_train=100, num_test=50)

trainingCNN(net=leNet, data_struct=data_struct, image_path='/home/mattpower/Downloads/Hands', palmar_dorsal='palmar', tot_exp=10)
trainingCNN(net=alexNet, data_struct=data_struct, image_path='/home/mattpower/Downloads/Hands', palmar_dorsal='dorsal', tot_exp=10)
print("Addestramento Reti Neurali Concluso")
streamEvaluation(net1=leNet, net2=alexNet, data_struct=data_struct, image_path='/home/mattpower/Downloads/Hands', tot_exp=10)
