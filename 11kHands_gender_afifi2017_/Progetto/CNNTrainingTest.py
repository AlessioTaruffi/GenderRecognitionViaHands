from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from MyLeNetCNN import MyLeNetCNN
from torch.utils.data import DataLoader
from CustomTransform import CustomDorsalTransform, CustomPalmTransform
from DataLoader import CustomImageDataset
from PrepareData import prepare_data


leNet = MyLeNetCNN(num_classes=2)

alexNet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)

# Update the final layer to output 10 classes
num_features = alexNet.classifier[6].in_features
alexNet.classifier[6] = nn.Linear(num_features, 2)

# Freeze all layers except the newly added fully connected layer
for param in alexNet.parameters():
    param.requires_grad = False
for param in alexNet.classifier[6].parameters():
    param.requires_grad = True

# Da mettere prima di richiamare le classi (reti)
data_struct = prepare_data(num_exp=10, num_train=200, num_test=400)

def trainingCNN(net:nn.Module, data_struct:dict, image_path:str, palmar_dorsal:str, tot_exp: int, batch_size=32, weight_decay=5e-05, learning_rate=0.001):
    # USIAMO LE NOSTRE :)
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
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for exp in range(tot_exp):
        #print(exp, data_struct[exp]['train']['palmar'])
        dataset_train = CustomImageDataset(image_dir=image_path, data_structure = data_struct, id_exp=exp, train_test='train', palmar_dorsal=palmar_dorsal, transform=[palmar_transform, dorsal_transform] )

        # Crea il DataLoader
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        net.train()
        running_loss = 0.0
        for _, data in enumerate(data_loader_train, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {exp + 1}, Loss: {running_loss / len(data_loader_train):.4f}')

    print('Finished Training')

#trainingCNN(net=leNet, data_struct=data_struct, image_path='/home/mattpower/Downloads/Hands', palmar_dorsal='palmar', tot_exp=10)
trainingCNN(net=alexNet, data_struct=data_struct, image_path='/home/mattpower/Downloads/Hands', palmar_dorsal='dorsal', tot_exp=10)

def testCNN(net:nn.Module, data_struct:dict, image_path:str, palmar_dorsal:str, tot_exp: int, batch_size=32):
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
    net.to(device)

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for exp in range(tot_exp):
            dataset_test = CustomImageDataset(image_dir=image_path, data_structure= data_struct, id_exp=exp, train_test='test', palmar_dorsal=palmar_dorsal, transform=[palmar_transform, dorsal_transform] )
            data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

            for data in data_loader_test:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test images: {100 * correct / total:.2f}%')


#testCNN(net=leNet, data_struct=data_struct, image_path='/home/mattpower/Downloads/Hands',  palmar_dorsal='palmar', tot_exp=10)
testCNN(net=alexNet, data_struct=data_struct, image_path='/home/mattpower/Downloads/Hands',  palmar_dorsal='dorsal', tot_exp=10)
