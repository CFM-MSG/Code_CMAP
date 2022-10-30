class AverageMeter(object):


  def __init__(self, precision=4):
    self.dph = "%.{}f".format(precision)
    self.epsilon = 1*10**(-precision)
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / (self.epsilon + self.count)

  def __str__(self):

    if self.count == 0:
      return str(self.val)

    return '{} ({})'.format(self.dph, self.dph) % (self.val, self.avg)