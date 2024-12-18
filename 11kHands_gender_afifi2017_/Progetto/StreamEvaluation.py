import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomTransform import CustomDorsalTransform, CustomPalmTransform
from CustomImageDataset import CustomImageDataset

def streamEvaluation(net1:nn.Module, net2:nn.Module, data_struct:dict, image_path:str, tot_exp: int, batch_size=32):
    # Definisci le trasformazioni da applicare alle immagini (opzionale)
    palmar_transform = transforms.Compose([
        CustomDorsalTransform(),
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

    tot_labels = torch.tensor([])
    tot_predicted = torch.tensor([])

    with torch.no_grad():
        for exp in range(tot_exp):
            dataset_dorsal = CustomImageDataset(image_dir=image_path, data_structure= data_struct, id_exp=exp, train_test='test', palmar_dorsal='dorsal', transform=[palmar_transform, dorsal_transform] )
            data_loader_dorsal = DataLoader(dataset_dorsal, batch_size=batch_size, shuffle=True)
           
            dataset_palmar = CustomImageDataset(image_dir=image_path, data_structure= data_struct, id_exp=exp, train_test='test', palmar_dorsal='palmar', transform=[palmar_transform, dorsal_transform] )
            data_loader_palmar= DataLoader(dataset_palmar, batch_size=batch_size, shuffle=True)

            for data_dorsal, data_palmar in zip(data_loader_dorsal, data_loader_palmar):
                
                dorsal_images, labels = data_dorsal
                dorsal_images, labels = dorsal_images.to(device), labels.to(device)

                palmar_images, labels = data_palmar
                palmar_images, labels = palmar_images.to(device), labels.to(device)
                
                # Softmax layer
                outputs_alexNetPalmar = net1(palmar_images)
                outputs_alexNetDorsal = net2(dorsal_images)

                # Score fusion layer
                #output_M = outputs_leNet[1] * 0.6  + outputs_alexNet[1] * 0.4
                #output_F = outputs_leNet[0] * 0.6  + outputs_alexNet[0] * 0.4

                # Applica la softmax agli output per ottenere le probabilità
                softmax = torch.nn.Softmax(dim=1)
                probs_alexNetPalmar = softmax(outputs_alexNetPalmar)
                probs_alexNetDorsal = softmax(outputs_alexNetDorsal)
    
                # Esegui la score fusion combinando le probabilità
                fused_probs = probs_alexNetPalmar * 0.7 + probs_alexNetDorsal * 0.3
    
                # Ottieni la previsione finale
                _, predicted = torch.max(fused_probs, 1)
                
                tot_labels = torch.cat((tot_labels, labels))
                tot_predicted = torch.cat((tot_predicted, predicted))

                '''
                # Etichette reali e predette
                y_true = [1, 0, 1, 1, 0, 1, 0, 0]  # Valori reali
                y_pred = [1, 0, 0, 1, 0, 1, 1, 0]  # Valori predetti

                # Calcolo della matrice di confusione
                cm = confusion_matrix(y_true, y_pred)

                print("Confusion Matrix:\n", cm)
                Significato della matrice:
                TP (Vero Positivo): 3 (le predizioni di classe "1" corrette)
                TN (Vero Negativo): 3 (le predizioni di classe "0" corrette)
                FP (Falso Positivo): 1 (ha predetto "1" quando il vero valore era "0")
                FN (Falso Negativo): 1 (ha predetto "0" quando il vero valore era "1")
                '''
    return tot_labels, tot_predicted
