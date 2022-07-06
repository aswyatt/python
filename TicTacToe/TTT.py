from multiprocessing.sharedctypes import Value
import numpy as np
import random

class TTT:
    def __init__(self) -> None:
        self.Grid = np.zeros((3, 3), dtype=np.int8)
        self.CurrentPlayer = 1

    def Play(self, row, col):
        if self.Grid[row, col]:
            raise ValueError(f"Invalid move: [{row}, {col}] is not empty")
        self.Grid[row, col] = self.CurrentPlayer
        self.CurrentPlayer = -self.CurrentPlayer

    def _CheckGrid(self, ax):
        S = T.Grid.sum(axis=ax)

    def Display(self):
        print(f"Current player: {self.CurrentPlayer}")
        print(self.Grid)
        print("")

def main():
    T = TTT()
    r = 0
    c = 1
    for n in range(1):
        while T.Grid[r, c]:
            r = random.randrange(3)
            c = random.randrange(3)
        T.Play(r, c)
        T.Display()

if __name__=="__main__":
    main()