import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Per creare il DataLoader devo prima creare la classe che eredita dal Dataset di Pytorch in cui definisco come dovranno essere caricate le immagini.
# In altre parole qui definisco le trasformazioni, unisco le immagini, inserisco il bordino, ecc, ecc
class CustomDataset(Dataset):
    def __init__(self, dataset_path, csv_path, separatorLen = 9, finalImageSize = 227): #, transform=None => Le trasformazioni le definisco direttamente qui, non fuori
        
        self.dataset_path = dataset_path
        self.csv_path = csv_path
        
        self.separatorLen = separatorLen
        self.finalImageSize = finalImageSize
        self.newSize = finalImageSize, int(finalImageSize/2 - separatorLen/2) # Voglio affiancare le due immagini ottenendo comunque una immagine quadrata.
        # Quindi imposto l'altezza a "228" e la larghezza a "228/2 - separatorPixel/2": in questo modo tolgo da ogni immagine la dimensione che avrà il separatore
        
        # Nel costruttore definisco una variabile con tutte le varie trasformazioni che avranno le immagini prima di essere unite
        self.firstTransform = transforms.Compose([ # Applicata alle singole immagini
            transforms.Resize(self.newSize), # Prende prima l'altezza poi la lunghezza
            transforms.Grayscale(num_output_channels=1),
        ])
        self.secondTransform = transforms.Compose([ # Applicata alle due immagini unite
            transforms.ToTensor(),
            transforms.Normalize((0.5633,), (0.1601,)) # Valori calcolati a parte
        ])
    
        # Leggi il file CSV e metti le info in variabile:
        self.csv_data = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.csv_data) # Restituisco la lunghezza del file CSV (equivale alla dimensione del DataSet)
    
    def __getitem__(self, index):
        # Prendo le immagini in base alle righe del file CSV aperto con Pandas
        image_name_1 = os.path.join(self.dataset_path, self.csv_data.iloc[index,0])
        image_name_2 = os.path.join(self.dataset_path, self.csv_data.iloc[index,1])
        
        
        # E' necessario che label rappresenti l'indice della classe corretta. Quindi 0 o 1 in formato Long (int64)
        label = self.csv_data.iloc[index, 2:]
        label = np.int64(label[0])
        
        
        # Apri le due immagini
        image_1 = Image.open(image_name_1)
        image_2 = Image.open(image_name_2)
        
        
        # Applica la prima trasformazione
        image_1 = self.firstTransform(image_1)
        image_2 = self.firstTransform(image_2)

        
        # Creo la nuova immagine composta (incluso il bordino)
        finalImage = Image.new('L', (image_1.width + image_2.width + self.separatorLen, image_1.height))
        finalImage.paste(image_1, (0, 0))
        finalImage.paste(image_2, (image_1.width + self.separatorLen, 0))
        
        finalImage = self.secondTransform(finalImage) # Trasformo in tensore e normalizzo
        
        
        row = {'finalImage': finalImage, 'label': label}
        
        return row