import datetime

from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
import matplotlib.pyplot as plt
from torchvision import transforms

datasetPath = "D:\\Università\\Machine Learning\\Progetto\\Dataset Manipulation\\Dataset\\"

csvPath = "D:\\Università\\Machine Learning\\Progetto\\Dataset Manipulation\\CSV\\"
csvTrainPath = csvPath + "csv_train.csv"

dataset_train = CustomDataset(dataset_path=datasetPath, csv_path=csvTrainPath)
loader_train = DataLoader(dataset_train, batch_size=1, num_workers=0, shuffle=False)

prox = next(iter(loader_train))["finalImage"][0]

plt.imshow(transforms.ToPILImage()(prox), cmap="gray")

print(datetime.datetime.now())

"""

# Come aveva fatto Ciccia:

label = self.csv_data.iloc[index, 2:]
label = np.array(label)
label = label.astype('int64') # Definisco il Label di questa coppia

#label = torch.Tensor(label)

# Fatto da Luca:
label = self.csv_data.iloc[index, 2:]
label = np.int64(label) # Nel modo di Ciccia ottengo una lista di numeri (?)
label = np.array(label)




print(prox.shape)

# NOTA: sto stampando una immagine normalizzata!! Andrebbe de-normalizzata

# Inverto la normalizzazione : soluzione trovata online
inv_normalize = transforms.Normalize(
   mean= [(-0.5633/0.1601)],
   std= [(1/0.1601)]
)

inv_tensor = inv_normalize(prox)
print(inv_tensor)

plt.imshow(transforms.ToPILImage()(inv_tensor), cmap="gray")

# , vmin=0, vmax=255
"""


"""
# Provo a caricare un'immagine
from PIL import Image

datasetPath = "D:\\Università\\Machine Learning\\Progetto\\Dataset Manipulation\\Dataset\\"

path1 = datasetPath + "1\\000000002.jpg"
path2 = datasetPath + "1\\000000010.jpg"

image1 = Image.open(path1)
image2 = Image.open(path2)

separatorPixels = 9 # Grandezza del separatore: dev'essere un numero dispari (affinché la newSize sia un formata da numeri interi). Provo impostando il doppio dello Stride (4) + 1

finalImageSize = 227 # 227x227 è la dimensione dell'immagine per cui AlexNet è stata creata

newSize = int(finalImageSize/2 - separatorPixels/2), finalImageSize # Voglio affiancare le due immagini ottenendo comunque una immagine quadrata.
# Quindi imposto l'altezza a "228" e la larghezza a "228/2 - separatorPixel/2": in questo modo tolgo da ogni immagine la dimensione che avrà il separatore

image1 = image1.convert('L') # converti image a gray scale
image2 = image2.convert('L')

image1 = image1.resize(newSize, Image.ANTIALIAS) # Ha dimensione 109 + 4.5 separatore
image2 = image2.resize(newSize, Image.ANTIALIAS) # Ha dimensione 109 + 4.5 separatore => Totale: 227x227

concat = Image.new('L', (image1.width + image2.width + separatorPixels, image1.height))
concat.paste(image1, (0, 0))
concat.paste(image2, (image1.width + separatorPixels, 0))

#concat.show() # Mostra l'immagine
# L'immagine va resa un TENSORE


# Soluzione trovata online di come calcolare Media e Std, ma è errata perché per la devStd fa la media delle devStd di ogni batch ("batch normalization")
"""

"""
for data in loader_train: #NOTA: data contiene i vari batch (restituisce quindi 512x1x227x227 se il batch_size impostato è 512)
    #print(data['finalImage'][0].shape)
    
    # Volendo mostrare l'immagine a video
    #img2 = transforms.ToPILImage(mode='L')(data['finalImage'][0])
    #plt.imshow(img2)
    #print(data['finalImage'].shape)
    
    batch_samples = data['finalImage'].size(0) # Restituirà 512 dato che è quello il Batch_size
    currImageTensor = data['finalImage'].view(batch_samples, data['finalImage'].size(1), -1)
    # Il precedente concatena in un unico array ogni riga di ogni foto.
    # Quindi si passa da 512x1x227x277 a 512x1x51529(ovvero 227x227)
    
    #print("currImageTensor.shape")
    #print(currImageTensor.shape)
    
    mean += currImageTensor.mean(2).sum(0) # Fa le medie di tutte le 512 foto del batch lungo la 2° dimensione (51529 elementi)
    
    #print("currImageTensor.mean(2).sum(0)")
    #print(currImageTensor.mean(2).sum(0))
    
    std += currImageTensor.std(2).sum(0) # Fa le std di tutte le 512 foto del batch lungo la 2° dimensione (51529 elementi)
    
    #print("currImageTensor.std(2).sum(0)")
    #print(currImageTensor.std(2).sum(0))
    
    nb_samples += batch_samples
    
    print("nb_samples")
    print(nb_samples)
    
    print("mean")
    print(mean)
    
    print("std")
    print(std)
    
    
mean /= nb_samples
std /= nb_samples

print("mean FINALE")
print(mean)

print("std FINALE")
print(std)
"""