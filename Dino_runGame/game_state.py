from main import *
'''
#%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from model import *
from PIL import ImageGrab
#import tensorflow
import cv2
import numpy as np
'''


def grab_screen(_driver=None):
    # bbox = region of interest on the entire screen
    screen = np.array(ImageGrab.grab(bbox=(40, 180, 440, 400))) #(left_x, top_y, right_x, bottom_y)

    image = process_img(screen)  # processing image as required

    return image


def process_img(image):
    # game is already in grey scale canvas, canny to get only edges and reduce unwanted objects(clouds)
    # resale image dimensions
    image = cv2.resize(image, (0, 0), fx=0.15, fy=0.10)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)

    #image.show()
    # crop out the dino agent from the frame
    image = image[2:48, 10:50]#image[2:38, 10:50]  # img[y:y+h, x:x+w]
    #cv2.imshow('image', image)
    #cv2.waitKey(0)

    image = cv2.Canny(image, threshold1=100, threshold2=200)  # apply the canny edge detection
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    return image


def get_state(actions, playerDino, cacti, pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter,
                   high_score, retbutton_image, gameover_image, HI_image, HI_rect,gameQuit):
    score = playerDino.score
    buffer_score = playerDino.score
    gameOver = False
    reward = 0.1 * score / 10  # dynamic reward calculation
    is_over = False  # game over
    if actions[1] == 1:  # else do nothing
        playerDino, cacti, pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter, high_score, gameOver = game_sysle(playerDino,
        cacti, pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter, high_score, True)
        #self._agent.jump()
        reward = 0.1 * playerDino.score / 11
    else:
        playerDino, cacti, pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter, high_score, gameOver = game_sysle(playerDino, cacti,
                    pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter,
                   high_score, False)
    image = grab_screen()

    if gameOver:
        # self._game.restart()
        reward = -11 / buffer_score
        print("\ngot ", buffer_score, "score before dead")
        #print("highest score", high_score, "\n")
        #restart_game()
        gameOver, gameQuit = game_over(highsc, retbutton_image, gameover_image, HI_image, HI_rect, gameOver, gameQuit)

        high_score, gamespeed, startMenu, gameOver, gameQuit, playerDino, new_ground, scb, highsc, counter, cacti, pteras, last_obstacle, HI_image, HI_rect, retbutton_image, retbutton_rect, gameover_image, gameover_rect = game_init()

        playerDino, cacti, pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter, high_score, gameOver = game_sysle(
            playerDino, cacti,
            pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter,
            high_score, True)
        is_over = True
    return image, reward, is_over, actions, playerDino, cacti, pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter, high_score, retbutton_image, gameover_image, HI_image, HI_rect,gameQuit


#screen = np.array(ImageGrab.grab(bbox=(40, 180, 440, 400)))  # (left_x, top_y, right_x, bottom_y)
#image = process_img(screen)  # processing image as required
#img = ImageGrab.grab(bbox=(50, 215, 550, 500)) #440, 400
#img = ImageGrab.grab(bbox=(45, 220, 445, 440))
#img.show()

#grab_screen()
