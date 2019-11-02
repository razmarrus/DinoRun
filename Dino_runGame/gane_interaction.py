#from main import *
import keras

class Game:
    def __init__(self, custom_config=True):
        self.score = 0
        self.over = False


    def get_score(self):
        return int(self.score)
