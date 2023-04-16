"""Flappy bird game based on https://youtu.be/MMxFDaIOHsE.

TODO: expand module description
"""
from typing import List
from utils import *
import pygame
import neat
import time
import os
import random

pygame.font.init()

# Load images (sprites / background etc.)
IMGS = {
    "Bird": [LoadImage(f"bird{n+1}.png") for n in range(3)],
    "Pipe": LoadImage("pipe.png"),
    "Base": LoadImage("base.png"),
    "BG": LoadImage("bg.png"),
}


class Bird:
    """Class to handle drawing and movement of bird.

    TODO: expand class description


    Attributes
    ----------
    IMGS
        List of pygame surfaces for each bird sprite
    MAX_ROTATION
        Maximum rotation angle of bird
    MIN_ROTATION
        Minimum rotation angle of bird
    DIVE_ROTATION
        When rotation angle of bird < this value, bird is considered to be diving (and will face towards the ground)
    ROT_VEL
        Rate of rotation of bird each frame
    ANIMATION_TIME
        TODO: check how this is used
    JUMP_VEL
        TODO: check how this is used
    VMAX
        TODO: check how this is used
    ACCEL
        TODO: check how this is used
    IMG_ORDER
        List of image indexes to use when animating bird flapping. Each frame will progress to next image index (cycling back to initial index)
    GLIDE_INDX
        Index of image to use when bird is diving

    x, y
        Current co-ordinates of bird.
        x>0 --> Right
        y>0 --> Up
    tilt
        Direction bird is facing.
        Increasing tilt rotates bird to face towards the sky (away from ground).
        Decreasing tilt rotates bird to face towards the ground.
    tick_count
        Number of time steps since last jump.
    vel
        Current vertical velocity of bird.
    height
        TODO: check usage. Current vertical height of bird.
    num_jumps
        Total number of "jump" key presses since start of current game.
    img_count
        Current image index used for sprite.
        Cycling through images creates flapping motion (or gliding).
    img
        Image currently displayed as the bird sprite.

    Methods
    -------
    jump
        Updates bird parameters when "jump" button pressed.
    move
        Implements physics engine for bird motion.
    draw(win)
        Updates bird sprite.
    get_mask
        TODO: ???
    """

    IMGS: List[pygame.surface.Surface] = IMGS["Bird"]
    MAX_ROTATION: int = 25
    MIN_ROTATION: int = -90
    DIVE_ROTATION: int = -60
    ROT_VEL: int = 15
    ANIMATION_TIME: int = 4
    JUMP_VEL: int = -4
    VMAX: int = 3
    ACCEL: float = 0.6
    IMG_ORDER: List[int] = [0, 1, 2, 1]
    GLIDE_INDX: int = 1

    def __init__(self, x: int, y: int) -> None:
        """
        Sets initial co-ordinates (x,y) of bird. Resets tilt, tick counter, velocity, height, sprite, number of jumps to 0.

        Parameters
        ----------
        x, y
            Initial co-ordinates of bird
        """

        self.x: int = x
        self.y: int = y
        self.tilt: int = 0
        self.tick_count: int = 0
        self.vel: int = 0
        self.height: int = self.y
        self.num_jumps: int = 0
        self.img_count: int = 0
        self.img: pygame.surface.Surface = self.IMGS[0]

    def jump(self) -> None:
        """Updates bird parameters when "jump" button pressed.

        Resets the tick counter, velocity and height of bird; zeroes tick counter and increments jump counter.
        """
        self.vel = self.JUMP_VEL
        self.tick_count = 0
        self.height = self.y
        self.num_jumps += 1

    def move(self) -> None:
        """Implements physics engine for bird motion.

        - Increments tick counter and adjusts position and velocity of bird (accelerates towards ground).
        - Rotates bird's facing direction towards ground (up to maximumum).
        """

        self.tick_count += 1
        self.y += self.tick_count * self.vel
        self.vel += (self.vel < self.VMAX) * self.ACCEL

        #   If moving up, point up, otherwise rotate down if not fully pointing down
        if self.vel < 0:
            self.tilt = self.MAX_ROTATION
        elif self.tilt > self.MIN_ROTATION:
            self.tilt -= self.ROT_VEL

    def draw(self, win: pygame.Surface) -> None:
        """Updates bird sprite.

        TODO:
        """

        #   If "diving", use glide-image otherwise cycle through images
        if self.tilt <= self.DIVE_ROTATION:
            img_indx = self.GLIDE_INDX
            self.img_count = (self.GLIDE_INDX + 1) * self.ANIMATION_TIME
        else:
            self.img_count += 1
            img_indx = (self.img_count // self.ANIMATION_TIME) % len(self.IMG_ORDER)
        self.img = self.IMGS[self.IMG_ORDER[img_indx]]
        rotated_img = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_img.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center
        )
        win.blit(rotated_img, new_rect.topleft)

    def get_mask(self) -> pygame.mask.Mask:
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 5
    PIPE_TOP = pygame.transform.flip(IMGS["Pipe"], False, True)
    PIPE_BOTTOM = IMGS["Pipe"]

    def __init__(self, x: int) -> None:
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.passed = False
        self.set_height()

    def set_height(self) -> None:
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP - random.randrange(50)

    def move(self) -> None:
        self.x -= self.VEL

    def draw(self, win: pygame.Surface) -> None:
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def CheckCollision(self, bird: Bird) -> bool:
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        y = round(bird.y)
        top_offset = (self.x - bird.x, self.top - y)
        bottom_offset = (self.x - bird.x, self.bottom - y)

        t_point = bird_mask.overlap(top_mask, top_offset)
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        return (t_point or b_point) is not None

    def CheckPassed(self, bird: Bird) -> bool:
        x = self.x + self.PIPE_TOP.get_width()
        self.passed = not self.passed and bird.x > x
        return self.passed

    def CheckRemove(self) -> bool:
        return self.x < -self.PIPE_TOP.get_width()


class ScrollingItem:
    def __init__(self, img: pygame.Surface, vel: int = 1, y: int = 0) -> None:
        self.IMG = img
        self.VEL = vel
        self.WIDTH = img.get_width()
        self.y = y
        self.x = [0, self.WIDTH]

    def move(self) -> None:
        W = self.WIDTH
        self.x = [(x + W - self.VEL) % (2 * W) - W for x in self.x]

    def draw(self, win: pygame.Surface) -> None:
        for x in self.x:
            win.blit(self.IMG, (x, self.y))


class Base(ScrollingItem):
    def __init__(self, y: int) -> None:
        ScrollingItem.__init__(self, IMGS["Base"], 5, y)

    def CheckCollision(self, bird: Bird) -> bool:
        return bird.y + bird.img.get_height() >= self.y


class Background(ScrollingItem):
    def __init__(self) -> None:
        ScrollingItem.__init__(self, IMGS["BG"], 1, 0)


class FlappyBird:
    FONT = pygame.font.SysFont("comicsans", 50)
    WIN_WIDTH = IMGS["BG"].get_width()
    WIN_HEIGHT = 800

    def __init__(self, NumBirds=1, KeyboardInput: bool = False, FPS: int = 30) -> None:
        self.win = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.KeyboardInput = KeyboardInput
        self.FPS = FPS
        self.reset(NumBirds)

    def reset(self, NumBirds=1) -> None:
        self.bird = [Bird(230, 350)] * NumBirds
        self.pipes = [Pipe(self.WIN_WIDTH + 20)]
        self.base = Base(self.WIN_HEIGHT - 70)
        self.BG = Background()
        self.score = 0

    def draw_window(self) -> None:
        # self.win.blit(IMGS["BG"], (0,0))
        self.BG.draw(self.win)
        for pipe in self.pipes:
            pipe.draw(self.win)
        text = self.FONT.render(f"Score: {self.score}", 1, (255, 255, 255))
        self.win.blit(text, (self.WIN_WIDTH - 10 - text.get_width(), 10))
        self.base.draw(self.win)
        self.bird.draw(self.win)
        pygame.display.update()

    def iterate(self) -> bool:
        self.BG.move()
        self.base.move()
        self.bird.move()

        rem = []
        add_pipe = False
        for pipe in self.pipes:
            failed = pipe.CheckCollision(self.bird)
            if pipe.CheckRemove():
                rem.append(pipe)
            add_pipe = pipe.CheckPassed(self.bird)
            pipe.move()

        if add_pipe:
            self.score += 1
            self.pipes.append(Pipe(self.WIN_WIDTH + random.randrange(200)))

        for r in rem:
            self.pipes.remove(r)
        self.draw_window()
        return failed or self.base.CheckCollision(self.bird)

    def run(self) -> None:
        run = True
        failed = False
        while run:
            self.clock.tick(self.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if self.KeyboardInput and event.type == pygame.KEYDOWN:
                    if failed and event.key == pygame.K_r:
                        self.reset()
                        failed = False
                    if event.key == pygame.K_q:
                        run = False
                    if event.key == pygame.K_SPACE:
                        self.bird.jump()
            if failed:
                continue
            failed = self.iterate()
            if failed:
                text = self.FONT.render("FAIL", 1, (150, 0, 0))
                x = (self.WIN_WIDTH - text.get_width()) // 2
                y = (self.WIN_HEIGHT - text.get_height()) // 2
                self.win.blit(text, (x, y))
                pygame.display.update()


def main():
    flappy = FlappyBird(True)
    flappy.run()
    pygame.quit()
    quit()


if __name__ == "__main__":
    main()
