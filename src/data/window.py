import numpy as np
import torch
from torch.utils.data import Dataset

#now we are creating a class that inherits from dataset , which is a standard class in pytorch used for holding and accessing data samples

class SlidingWindowDataset(Dataset):#this class is defined and indicates that it inherits Dataset base class
    def __init__(self, metrics, events, labels, T=30):
        #self: the instace that it is created
        #metrics: machine performance metrics
        #events: event that counts like failures and evictions
        #labels: the target value for the predictions
        # T is the window size used for each example

        """
        metrics: np.ndarray(N, Fm) sorted by time within each machine
        events: np.ndarray(N, Fe)
        LABELS: np.ndarray(N,) This gives a label at each time step for horizon h
        We assume pre-grouped into contious sequences per machine
        :param metrics:
        :param events:
        :param labels:
        :param T:
        """

        self.xm, self.xe, self.y=[], [], []
        #here on the above line this initializes three empty lists
        #and each tores windows and labels

        for seq_m, seq_e, seq_y in zip(metrics, events, labels):
            #the above line is used to iterate in the input data like metrics, events and labels
            for t in range(T-1, len(seq_m)):
                self.xm.append(seq_m[t - T + 1:t + 1])
                self.xe.append(seq_e[t - T + 1:t + 1])
                self.y.append(seq_y[t])
            self.xm = torch.tensor(np.stack(self.xm), dtype=torch.float32)#converting it into a pytorch tensors (metrics)
            self.xe = torch.tensor(np.stack(self.xe), dtype=torch.float32)#converting to pytorch tensors (events)
            self.y = torch.tensor(np.array(self.y), dtype=torch.float32)#labels


    def __len__(self):
        return len(self.y) #returns the size of the final label tensor

    def __getitem__(self, i):
        return self.xm[i], self.xe[i], self.y[i]
    #the above function returns the i th value or sample and that has three parts and those are ..
    #metrics window , event window and corresponding label

