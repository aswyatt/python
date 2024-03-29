import numpy as np
import random

class TTT:
    def __init__(self) -> None:
        self.Reset()

    def Reset(self):
        self.Grid = np.zeros((3, 3), dtype=np.int8)
        self.CurrentPlayer = 1
        self.Turn = 1

    def Play(self, row:int, col:int) -> None:
        if self.Turn>9:
            raise ValueError("Game is draw - please reset")
        if self.Grid[row, col]:
            raise ValueError(f"Invalid move: [{row}, {col}] is not empty")
        self.Grid[row, col] = self.CurrentPlayer
        self.CurrentPlayer = -self.CurrentPlayer
        self.Turn += 1

    def CheckForWinner(self) -> int:
        G = self.Grid
        D = G.diagonal().reshape((-1, 1))
        AD = np.fliplr(G).diagonal().reshape((-1,1))
        S = np.hstack((G, G.T, D, AD)).sum(0)
        S = G.sum(0)
        if self.Turn>9:
            return 0
        elif np.any(S==3):
            return 1
        elif np.any(S==-3):
            return -1
        else:
            return None

    def Display(self):
        print(f"Current player: {self.CurrentPlayer}")
        print(self.Grid)
        print("")

def main():
    T = TTT()
    r = 0
    c = 1
    W = None
    while W is None:
        while T.Grid[r, c]:
            r = random.randrange(3)
            c = random.randrange(3)
        T.Play(r, c)
        T.Display()
        W = T.CheckForWinner()
    if W==0:
        print("Draw")
    else:
        print("Winner is player", W)

if __name__=="__main__":
    main()