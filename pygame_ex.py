import pygame, sys, random
from pygame.locals import *

arr=[]
RED=(255,10,10)
BLU=(10,255,10)
GRN=(10,10,255)
arr.append(RED)
arr.append(BLU)
arr.append(GRN)
SCREEN_X=100
SCREEN_Y=100

screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y))
square=pygame.Surface((1, 1))

for j in range (SCREEN_Y):
    for i in range (SCREEN_X):
        num=random.randint(0,2)
        square.fill(arr[num])
        draw_me=pygame.Rect((j+1), (i+1), 1, 1)
        screen.blit(square,draw_me)

running = True
while running: # This would start the event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # This would be a quit event.
            running = False # So the user can close the program
    pygame.display.flip()

pygame.quit()                        
