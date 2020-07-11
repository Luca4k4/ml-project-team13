import numpy as np

class MeanStdCalculator():
    def __init__(loader_train):
        self.loader_train = loader_train
    
    def computeMean():
        somma = 0.
        media = 0.
        
        numPixel = 0
        countSamples = 0 # usato per capire a che punto è arrivato
        
        for data in loader_train:
            countSamples += data['finalImage'].size(0) # Restituisce il # di elementi in questo batch (utile per capire dove è arrivato)
            print("Immagini fatte (MEDIA): ")
            print(countSamples)
            
            reshaped = np.reshape(data['finalImage'], -1) # Pongo tutto il batch come unico array
            
            numPixel += reshaped.size(0)
            somma += reshaped.sum(0)
            
            print("numPixel computati: ")
            print(numPixel)
            print("somma corrente: ")
            print(somma)
            
            #currImageTensor = data['finalImage'].view(batch_samples, data['finalImage'].size(1), -1)
            # Il precedente concatena in un unico array ogni riga di ogni foto.
            # Quindi si passa da 512x1x227x277 a 512x1x51529(ovvero 227x227)
            
            #somma += currImageTensor.sum(2)
            #print(somma)
            #numPixel += currImageTensor.size(2)
        
        media = somma/numPixel
        print("MEDIA FINALE: ")
        print(media)
        
        return media
        
    def computeStd(mean):
        sommaDevStd = 0.
        devStd = 0.
        
        countSamples = 0 # usato per capire a che punto è arrivato
                
        for data in loader_train:
            countSamples += data['finalImage'].size(0) # Restituisce il # di elementi in questo batch (utile per capire dove è arrivato)
            print("Immagini fatte (DEV.STD): ")
            print(countSamples)
            
            reshaped = np.reshape(data['finalImage'], -1) # Pongo tutto il batch come unico array
            sommaDevStd += ((reshaped-media)**2).sum()
            
            print("sommaDevStd corrente:")
            print(sommaDevStd)
            
        
        devStd = np.sqrt(sommaDevStd/numPixel)
        print("DEV. STANDARD FINALE: ")
        print(devStd)
        
        return devStd
    
    def getMeanAndStd():
        media = self.computeMean()
        devStd = self.computeStd(media)
        
        print("MEDIA FINALE: ")
        print(media)
        
        print("DEV. STANDARD FINALE: ")
        print(devStd)
        
        return {"media": media, "devStd": devStd}
        
        
        