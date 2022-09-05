from cmath import inf
import pygame
import os

PATH = os.path.dirname(__file__)

def LoadImage(file: str):
    return pygame.transform.scale2x(
        pygame.image.load(os.path.join(PATH, "imgs", file))
        )

def Clamp(val, minval=-inf, maxval=inf):
    return max(min(maxval, val), minval)