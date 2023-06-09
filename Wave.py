class Wave:
    def __init__(self, min1, max, min2, T):
        self.min1 = min1
        self.max = max
        self.min2 = min2
        self.T = T * 10 / 60 # т.е. в минутах

    def height(self):
        return (2*self.max - self.min1 - self.min2) / 2
