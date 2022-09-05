#   Flappy bird game based on https://youtu.be/MMxFDaIOHsE
from utils import *
import pygame
import neat
import time
import os
import random

#   Load images
IMGS = {
    "Bird": [LoadImage(f"bird{n+1}.png") for n in range(3)],
    "Pipe": LoadImage("pipe.png"),
    "Base": LoadImage("base.png"),
    "BG": LoadImage("bg.png")
}


class Bird:
    IMGS = IMGS["Bird"]
    MAX_ROTATION = 25
    MIN_ROTATION = -90
    DIVE_ROTATION = -80
    ROT_VEL = 10
    ANIMATION_TIME = 5
    JUMP_VEL = -10.5
    IMG_ORDER = [0, 1, 2, 1]
    GLIDE_INDX = 1

    def __init__(self, x:int, y:int) -> None:
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self) -> None:
        self.vel = self.JUMP_VEL
        self.tick_count = 0
        self.height = self.y

    def move(self) -> None:
        self.tick_count += 1
        dy = Clamp(self.tick_count*(self.vel + 1.5*self.tick_count), -2, 16)
        self.y += dy

        #   Could be achieved using logic multiplication only
        #   If moving up or above initial height, point up, otherwise tilt down if not pointing down
        if dy<0 or self.y<self.height+50:
            self.tilt = self.MAX_ROTATION
        elif self.tilt>self.MIN_ROTATION:
            self.tilt -= self.ROT_VEL

    def draw(self, win:pygame.Surface) -> None:
        #   If "diving", use glide-image otherwise cycle through images
        if self.tilt<=self.DIVE_ROTATION:
            img_indx = self.GLIDE_INDX
            self.img_count = (self.GLIDE_INDX+1)*self.ANIMATION_TIME
        else:
            self.img_count += 1
            img_indx = (self.img_count//self.ANIMATION_TIME) % len(self.IMG_ORDER)
        self.img = self.IMGS[self.IMG_ORDER[img_indx]]
        rotated_img = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_img.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center
        )
        win.blit(rotated_img, new_rect.topleft)

    def get_mask(self) -> pygame.mask.Mask:
        return pygame.mask.from_surface(self.img)


class FlappyBird:
    WIN_WIDTH = 500
    WIN_HEIGHT = 800
    FPS = 30

    def __init__(self) -> None:
        self.win = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.bird = Bird(200,200)

    def draw_window(self) -> None:
        self.win.blit(IMGS["BG"], (0,0))
        self.bird.draw(self.win)
        pygame.display.update()

    def run(self) -> None:
        run = True
        while run:
            self.clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
            self.bird.move()
            self.draw_window()

def main():
    flappy = FlappyBird()
    flappy.run()
    pygame.quit()
    quit()

if __name__=="__main__":
    main()