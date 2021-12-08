import numpy as np

class LossContainer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = []
        self.classloss = []
        self.sampleloss = []
    
    def append(self, loss_batch, classloss_batch=None, sampleloss_batch=None):
        self.loss.append(loss_batch.data.item())
        self.classloss.append(classloss_batch)
        self.sampleloss.append(sampleloss_batch)

    def calc_mean(self):
        tup = ()

        tup = tup + (np.mean(self.loss),) if self.loss else tup + (0, )
        tup = tup + (np.mean(self.classloss, 0),) if self.classloss else tup + (0,)
        return tup
