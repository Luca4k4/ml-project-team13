import torch
from torch import nn # Importo il modulo di PyTorch per le reti neurali
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from AverageValueMeter import AverageValueMeter
from sklearn.metrics import accuracy_score
from os.path import join
import numpy as np

import datetime

# def train_classifier(model, train_loader, test_loader, exp_name='general', lr=0.01, epochs=10, momentum=0.99, logdir='logs'): 
def train_classifier(model, train_loader, test_loader, exp_name, lr, epochs, momentum, logdir):
    
  print("Data e ora (INIZIO):")
  print(datetime.datetime.now()) # In realtà passando il mouse sopra la curva su TensorBoard, compare il tempo
    
  criterion = nn.CrossEntropyLoss() # Il Criterion implementa il SoftMax con la Loss CrossEntropy
  optimizer = SGD(model.parameters(), lr, momentum=momentum) # L'Optimizer contiene le funzioni per fare la GD facilmente
  
  # meters: Servono a disegnare un grafico pulito nel caso della SGD: se loggassi la loss ad ogni batch il risultato sarebbe falsato (il num di elementi per batch cambia)
  # Quindi uso AverageValueMeter() che tiene i valori aggiornati man mano e poi stampa la media corretta
  loss_meter = AverageValueMeter()
  acc_meter = AverageValueMeter()
  
  # writer per la scrittura dei Log: imposto il writer a stampare nella cartella passata in input
  writer = SummaryWriter(join(logdir, exp_name))
  
  # device: se Cuda non è Available, eseguo i calcoli sulla CPU
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  # definiamo un dizionario contenente i loader di training e test
  loader = {
  'train' : train_loader,
  'test' : test_loader
  }

  # inizializziamo il global step: calcola i valori già 'macinati'
  global_step = 0
  
  for e in range(epochs):
    # iteriamo tra due modalità: train e test
    for mode in ['train','test']:
      
      loss_meter.reset(); acc_meter.reset() # Imposto a 0 somma e numeri dei due AverageValueMeter
      model.train() if mode == 'train' else model.eval() # Dico al modello se sono in modalità allenamento o valutazione (lui migliora le performance in base alla scelta)
      
      with torch.set_grad_enabled(mode=='train'): # Abilitiamo i gradienti solo in training (altrimenti l.backward non funzionerebbe)
        # Il costrutto 'with' dice che deve disabilitare il gradiente SOLO per le righe seguenti (identate) => Questo è importante sia per evitare cose strane (es. altre backward chiamate su test/validation set) sia per risparmiare risorse computazionali
        provapp = 1
        for i, batch in enumerate(loader[mode]): # Ciclo eseguito 2 volte: una per la fase di training, una per quella di test
          provapp += 1
          # Prendo dal dataloader N immagini ed N etichette: N dipende dalla grandezza del batch 
          x = batch["finalImage"].to(device) # Utilizzo la stessa voce "finalImage" presente nel CustomDataset
          y = batch["label"].to(device) # Idem a sopra. Etichette reali. Questi vengono portati sul device corretto (in genere GPU)
          
          output = model(x) # Output contiene le N etichette predette dalla rete (ognuna ha 2 classi dato che il classificatore è binario)
          
          """
          print("output:")
          print(output)
          
          print("y:")
          print(y)
          """
          
          # aggiorniamo il global_step => conterrà il numero di campioni visti durante il training
          n = x.shape[0] # numero di elementi nel batch ('N')
          global_step += n
                     
          # Calcoliamo la Loss 
          # NOTA IMPORTANTE: il secondo elemento ("y") è l'array degli indici della classe corretta in formato Long, non è il tensore delle classi! y ha dimensione N
          l = criterion(output,y)          
          
          if mode=='train':
            l.backward() # Questa funzione scende il gradiente di un passo in maniera totalmente automatica (calcola quindi le derivate)
            optimizer.step()
            optimizer.zero_grad()
        
          """
          print("Massimo pixel dell'immagine")
          print(x.max())

          print("Indice della classe Corretta:")
          print(y)
          
          print("Valori predetti dove verrà visto max indice")
          print(output.to('cpu'))
          
          print("riga CSV")
          print(provapp)
          
          
          tepp = 0
          for param in model.parameters():
              print("parametro")
              print(param)
              break # stampa i parametr
          
          #return # solo per test
          """
          
          acc = accuracy_score(y.to('cpu'), output.to('cpu').max(1)[1]) # Il secondo parametro in pratica cerca il massimo, lo pone ad 1 e restituisce il suo indice (0 o 1 dato che le classi sono 2)
          loss_meter.add(l.item(),n)
          acc_meter.add(acc,n)

          #logghiamo i risultati iterazione per iterazione solo durante il training
          if mode=='train':
            writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
            writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)
            
      #una volta finita l'epoca (sia nel caso di training che test, loggiamo le stime finali)
      writer.add_scalar('loss/' + mode, loss_meter.value(), global_step=global_step)
      writer.add_scalar('accuracy/' + mode, acc_meter.value(), global_step=global_step)

    #conserviamo i pesi del modello alla fine di un ciclo di training e test
    torch.save(model.state_dict(),'%s-%d.pth'%(exp_name,e+1))
    
    print("Epoca finita: ")
    print(e)
    
    print("Data e ora (FINE EPOCA):")
    print(datetime.datetime.now())
    
    print("Campioni analizzati: ")
    print(global_step)
  return model


