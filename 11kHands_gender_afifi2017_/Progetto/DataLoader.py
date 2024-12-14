from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import Dataset

# Classe per effettuare la creazione del dataset utilizzato per il training e test
class CustomImageDataset(Dataset):
    def __init__(self, data_structure, image_dir, id_exp, train_test, palmar_dorsal, transform=None):
        """
            image_dir (str): Percorso della cartella contenente le immagini.
            json_file (str): Percorso del file JSON contenente le etichette.
        """
       # self.labels = 
        if palmar_dorsal == 'dorsal':
            self.image_filenames = data_structure[id_exp][train_test]['images'][1]
        else: 
            self.image_filenames = data_structure[id_exp][train_test]['images'][0]
        print(len(self.image_filenames), len(data_structure[id_exp][train_test]['labels']))
        self.labels = dict(zip(self.image_filenames , data_structure[id_exp][train_test]['labels']))
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
        # Ottieni il nome dell'immagine
        img_name = self.image_filenames[idx]  
        # Costruisci il percorso completo dell'immagine
        img_path = os.path.join(self.image_dir, img_name)  
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