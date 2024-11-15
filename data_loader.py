import numpy as np

class epileptor:
    def __init__(self, path: str):
        self.name = "epileptor parameters data loader"
        self.path = path

    def __str__(self):
        return f"{self.name} with data in {self.path}."
    
    def load(self, x0_filename: str, CpES_filename: str):
        
        self.x0 = np.load(self.path + x0_filename)
        self.CpES = np.load(self.path + CpES_filename)

        return self.x0, self.CpES