# Definiamo la classe AverageValueMeter già definita nello scorso laboratorio,
# Serve per tenere una media aggiornata durante il Training
class AverageValueMeter():
  def __init__(self):
    self.reset()

  def reset(self):
    self.sum = 0
    self.num = 0

  def add(self, value, num):
    self.sum += value*num
    self.num += num

  def value(self):
    try:
      return self.sum/self.num
    except:
      return None