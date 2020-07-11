# Creo la CNN AlexNet:
# Fonte da cui ho studiato la rete: https://andreaprovino.it/alexnet/

from torch import nn # Importo il modulo di PyTorch per le reti neurali

class AlexNet(nn.Module):
  def __init__(self, input_channels=1, out_classes=2):
    super(AlexNet, self).__init__()

    # Ridefiniamo il modello utilizzando i moduli sequential.
    # Ne definiamo due: un "feature extractor", che estrae le feature maps e un "classificatore" che implementa i livelli FC
    self.feature_extractor = nn.Sequential(
        
      # Per calcolare le dimensioni dell'output della convoluzione: [(W−K+2P)/S]+1 dove:
      # - W è il volume dell'input
      # - K è la dimensione del kernel
      # - P Padding e S stride

      # L'immagine in input ha dimensione 227x227 (finalImageSize)

      # Convoluzione 1
      nn.Conv2d(input_channels, 96, 11, stride=4), # Realizza 96 kernel di dimensioni 11x11 con uno stride 4 => Output: 55x55x96 (96 sono i kernel, 55 è il risultato dell'operazione fatta prima)
      nn.MaxPool2d(3, stride=2), # Riduce le dimensioni di un fattore 3 (applica kernel 3x3) e ha uno stride di 2 => Output: 27x27x96 (sempre utilizzando la formula scritta sopra...)
      nn.ReLU(), # Applico la ReLu

      # Convoluzione 2
      nn.Conv2d(96, 256, 5, stride=1, padding=2), # Output: 27x27x256
      nn.MaxPool2d(3, stride=2), # Output: 13x13x256
      nn.ReLU(),

      # Convoluzione 3
      nn.Conv2d(256, 384, 3, stride=1, padding=1), # Output: 13x13x384
      nn.ReLU(),
      # NOTA: in alcuni livelli, il MaxPooling non viene fatto, questo per evitare che le mappe diventino troppo piccole

      # Convoluzione 4
      nn.Conv2d(384, 384, 3, stride=1, padding=1), # Output: 13x13x384
      nn.ReLU(),
      
      # Convoluzione 5
      nn.Conv2d(384, 256, 3, stride=1, padding=1), # Output: 13x13x256
      nn.MaxPool2d(3, stride=2), # Output: 6x6x256
      nn.ReLU()
    )

    self.classifier = nn.Sequential(
      # Fully Connected 6
      nn.Linear(9216, 4096), #Input: 6x6x256=9216 => 4096 in output
      nn.ReLU(),

      # Fully Connected 7
      nn.Linear(4096, 4096),
      nn.ReLU(),

      # Fully Connected 8
      nn.Linear(4096, out_classes)
    )

  def forward(self,x):
    #Applichiamo le diverse trasformazioni in cascata
    x = self.feature_extractor(x)
    x = self.classifier(x.view(x.shape[0],-1))
    return x


# DA CALCOLARE => Il numero di parametri da imparare