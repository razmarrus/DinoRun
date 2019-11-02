#model hyper parameters
from collections import deque
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from matplotlib import pyplot as plt

loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
scores_file_path = "./objects/scores_df.csv"

#import seaborn as sns

loss_df = pd.read_csv(loss_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns =['loss'])
scores_df = pd.read_csv(scores_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns = ['scores'])
actions_df = pd.read_csv(actions_file_path) if os.path.isfile(actions_file_path) else pd.DataFrame(columns = ['actions'])

ACTIONS = 2 # possible actions: jump, do nothing
GAMMA = 0.99 # decay rate of past observations original 0.99
OBSERVATION = 50000. # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows , img_cols = 40,20
img_channels = 4 #We stack 4 frames




def save_obj(obj, name ):
    with open('objects/'+ name + '.pkl', 'wb') as f: #dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def init_cache():
    """initial variable caching, done only once"""
    save_obj(INITIAL_EPSILON,"epsilon")
    t = 0
    save_obj(t,"time")
    D = deque()
    save_obj(D,"D")

'''
def show_plots():
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(15, 15))
    axs[0].set_title('Loss')
    axs[1].set_title('Game Score progress')
    #loss_df = pd.read_csv("./objects/loss_df.csv").clip(0, 50).tail(100000)
    scores_df = pd.read_csv("./objects/scores_df.csv").head(190000)

    actions_df = pd.read_csv("./objects/actions_df.csv").tail(100000)
    #loss_df['loss'] = loss_df['loss'].astype('float')
    #loss_df.plot(use_index=True, ax=axs[0])
    #scores_df.plot(ax=axs[1])
    #     sns.distplot(actions_df,ax=axs[2])
    imgg = fig.canvas.draw()
    img.show()


show_plots()
'''

