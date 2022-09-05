#   Flappy bird game based on https://youtu.be/MMxFDaIOHsE
from utils import *
import pygame
import neat
import time
import os
import random

#   Define size of window
WIN_WIDTH = 500
WIN_HEIGHT = 800

#   Load images
IMGS = {
    "Bird": [LoadImage(f"bird{n+1}.png") for n in range(3)],
    "Pipe": LoadImage("pipe.png"),
    "Base": LoadImage("base.png"),
    "BG": LoadImage("bg.png")
}