from main import *
from game_state import *
from collections import *
from time import *


def buildmodel():
    print("Now we build the model")
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(img_cols,img_rows,img_channels)))  #20*40*4
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(ACTIONS))
    adam = keras.optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model


def trainNetwork(model):

    def trainBatch(minibatch):
        inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 20, 40, 4
        targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2
        loss = 0
        D = load_obj("D")  # load from file system

        for i in range(0, len(minibatch)):
            state_t = minibatch[i][0]  # 4D stack of images
            action_t = minibatch[i][1]  # This is action index
            reward_t = minibatch[i][2]  # reward at state_t due to action_t
            state_t1 = minibatch[i][3]  # next state
            terminal = minibatch[i][4]  # wheather the agent died or survided due the action
            inputs[i:i + 1] = state_t
            targets[i] = model.predict(state_t)  # predicted q values
            Q_sa = model.predict(state_t1)  # predict q values for next step
            if terminal:
                targets[i, action_t] = reward_t  # if terminated, only equals reward
            else:
                targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)
            #loss_df.loc[len(loss_df)] = loss
            return loss, Q_sa
    # store the previous observations in replay memory
    #D =  deque()  # load from file system
    # get the first state by doing nothing
    D = load_obj("D")
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1  # 0 => do nothing,
    # 1=> jump
    high_score, gamespeed, startMenu, gameOver, gameQuit, playerDino, new_ground, scb, highsc, counter, cacti, pteras, last_obstacle, HI_image, HI_rect, retbutton_image, retbutton_rect, gameover_image, gameover_rect = game_init()
    #x_t, r_0, terminal = game_state.get_state(do_nothing)  # get next step after performing the action

    x_t, r_0, terminal, actions, playerDino, cacti, pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter, high_score, retbutton_image, gameover_image, HI_image, HI_rect,gameQuit = get_state(do_nothing,
            playerDino, cacti, pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter,
                   high_score, retbutton_image, gameover_image, HI_image, HI_rect,gameQuit)

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2).reshape(1, 20, 40, 4)  # stack 4 images to create placeholder input reshaped 1*20*40*4

    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON
    t = 0
    prev_score_printer = 0
    max_score = 0
                      #We go to training mode

    OBSERVE = OBSERVATION
    epsilon = load_obj("epsilon")
    model.load_weights("model_final.h5")
    adam = keras.optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)

    t = load_obj("time") # resume from the previous time step stored in file system


    while (t < 1000001):  # endless running

        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0  # reward at t
        a_t = np.zeros([ACTIONS])  # action at t

        # choose an action epsilon greedy
        if random.random() <= epsilon:  # randomly explore an action
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:  # predict the output
            q = model.predict(s_t)  # input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)  # chosing index with maximum q value
            action_index = max_Q
            a_t[action_index] = 1  # o=> do nothing, 1=> jump

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observed next state and reward
        x_t1, r_t, terminal, actions, playerDino, cacti, pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter, high_score, retbutton_image, gameover_image, HI_image, HI_rect,gameQuit = get_state(a_t,
            playerDino, cacti, pteras, last_obstacle, gamespeed, new_ground, scb, highsc, counter,
                   high_score, retbutton_image, gameover_image, HI_image, HI_rect,gameQuit)
        #last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x20x40x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)  # append the new image to input stack and remove the first one

        if playerDino.score % 5 == 0 and prev_score_printer != playerDino.score:
            print("Currrent score", playerDino.score)
            prev_score_printer = playerDino.score
            if playerDino.score > max_score:
                max_score = playerDino.score
        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing; sample a minibatch to train on
        if t > OBSERVE:
            loss,  Q_sa = trainBatch(random.sample(D, BATCH))

        s_t = s_t1
        t = t + 1
        # save progress every 1000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            print("highest score", max_score)

            model.save_weights("model_final.h5", overwrite=True)
            save_obj(D, "D")  # saving episodes
            save_obj(t, "time")  # caching time steps
            save_obj(epsilon, "epsilon")  # cache epsilon to avoid repeated randomness in actions
            #loss_df.to_csv("./objects/loss_df.csv", index=False)
            scores_df.to_csv("./objects/scores_df.csv", index=False)
            actions_df.to_csv("./objects/actions_df.csv", index=False)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if t % 20 == 0:
            print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX ", np.max(Q_sa),
                  "/ Loss ", loss)


def playGame(observe=False):

    model = buildmodel()
    trainNetwork(model)


playGame()