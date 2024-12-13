from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PrepareData import prepare_data
from CustomTransform import CustomPalmTransform, CustomDorsalTransform

class CustomImageDataset(Dataset):
    def __init__(self, data_structure, image_dir, id_exp, train_test, palmar_dorsal, transform=None):
        """
        Args:
            image_dir (str): Percorso della cartella contenente le immagini.
            json_file (str): Percorso del file JSON contenente le etichette.
            transform (callable, optional): Trasformazioni da applicare sulle immagini.
            {
    "image1.jpg": 0,
    "image2.jpg": 1,
    "image3.jpg": 2 }
        """
        
        self.labels = dict()

        male_female_list = ['male', 'female']
        # Lista dei nomi dei file immagine
        #print(data_structure)
        #print(id_exp)
        #print(train_test)
        #print(palmar_dorsal)

        self.image_filenames = data_structure[id_exp][train_test][palmar_dorsal]['male'] + data_structure[id_exp][train_test][palmar_dorsal]['female'] 

        for gender in male_female_list:
            for name in data_structure[id_exp][train_test][palmar_dorsal][gender]:
                self.labels[name] = 0 if gender == 'male' else 1

        self.image_dir = image_dir
      

        # Da cambiare
        self.id_exp = id_exp
        self.train_test = train_test
        self.palmar_dorsal = palmar_dorsal
        self.transform = transform

    def __len__(self):
        """Restituisce il numero di campioni nel dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Restituisce una coppia (immagine, etichetta) data l'indice."""
        img_name = self.image_filenames[idx]  # Ottieni il nome dell'immagine
        img_path = os.path.join(self.image_dir, img_name)  # Costruisci il percorso completo dell'immagine
        
        # Carica l'immagine
        image = Image.open(img_path).convert("RGB")
        
        # Ottieni l'etichetta dal JSON
        label = self.labels[img_name]
        
        # Applica le trasformazioni se fornite
        if self.palmar_dorsal == 'palmar':
            image = self.transform[0](image)
        else: 
            image = self.transform[1](image)
        return image, label


'''
for data, target in iter(data_loader):
    print(f"Immagini: {data.shape}")  # Stampa la forma delle immagini (batch_size, canali, altezza, larghezza)
    print(f"Etichette: {target}")  # Stampa le etichette del batch
    # Uscita dal ciclo dopo il primo batch (puoi rimuovere questa linea se vuoi vedere più batch)
    break


# Iterazione attraverso il DataLoader
for images, labels in data_loader:
    print(images.shape)  # Dimensione del batch di immagini
    print(labels.shape)  # Dimensione delle etichette
    break  # Esci dopo il primo batch


# Iterazione
for batch_idx, (x, y) in enumerate(data_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Dati: {x.shape}, Etichette: {y.shape}")
    break
'''

