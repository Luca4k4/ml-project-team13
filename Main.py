# Eseguo gli import:
import torch
import numpy as np
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from AlexNet import AlexNet
# from MeanStdCalculator import MeanStdCalculator # Valori già calcolati, non serve ricalcolarli per adesso
from Classifier import train_classifier # Importo la sola funzione (non è necessario che sia una Classe)

# Testo di essere in GPU:
# print(torch.cuda.is_available())

# Imposto i seed per rendere l'esperimento replicabile:
np.random.seed(2109)
torch.random.manual_seed(2109);

# ======================= Constanti, vanno definite in un file a parte (già creato dai colleghi) ================
datasetPath = "D:\\Università\\Machine Learning\\Progetto\\Dataset Manipulation\\Dataset\\"

csvPath = "D:\\Università\\Machine Learning\\Progetto\\Dataset Manipulation\\CSV\\"
csvTrainPath = csvPath + "csv_train.csv"
csvTestPath = csvPath + "csv_test.csv"
csvValidationPath = csvPath + "csv_validation.csv"

#logsDir = "D:\\Università\\Machine Learning\\Progetto\\AlexNet\\logs"
logsDir = "logs"

batchSize = 64

# =======================================================================
# COMANDI DA DARE IN CONSOLE PER APRIRE TensorBoard

# %load_ext tensorboard
# %tensorboard --logdir "logs"
# %tensorboard --logdir=logs/ --host localhost --port 8088 <====

# ANDARE POI DA BROWSER A: http://localhost:6006/
# ANDARE POI DA BROWSER A: http://localhost:8088/ <====

# =======================================================================

# ============================ Creo i DataLoader ===============================
# Definiamo il DataLoader (quello che dice come caricare il Dataset) di Train, di Test e di Validation con un batch_size di X immagini

dataset_train = CustomDataset(dataset_path=datasetPath, csv_path=csvTrainPath)
loader_train = DataLoader(dataset_train, batch_size=batchSize, num_workers=0, shuffle=False)
# SHUFFLE a False perché le immagini sono già rimescolate nel CSV

dataset_test = CustomDataset(dataset_path=datasetPath, csv_path=csvTestPath)
loader_test = DataLoader(dataset_test, batch_size=batchSize, num_workers=0, shuffle=False)

dataset_validation = CustomDataset(dataset_path=datasetPath, csv_path=csvValidationPath)
loader_validation = DataLoader(dataset_validation, batch_size=batchSize, num_workers=0, shuffle=False)

# =======================================================================

# def train_classifier(model, train_loader, test_loader, exp_name='general', lr=0.01, epochs=10, momentum=0.99, logdir='logs'): 
# A questo punto istanziamo il modello e vediamo come si comporta
alexnet = AlexNet()

# NOTA IMPORTANTE: con un learning rate troppo alto, il grafico su Tensorboard va all'infinito (problema già con lr=0.01)!!
alexnet_trained = train_classifier(alexnet, loader_train, loader_test, 'alexnet', lr=0.002, epochs=150, momentum=0.8, logdir=logsDir)