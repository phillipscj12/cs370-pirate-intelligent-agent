#!/usr/bin/env python
# coding: utf-8

# # Treasure Hunt Game Notebook
# 
# ## Read and Review Your Starter Code
# The theme of this project is a popular treasure hunt game in which the player needs to find the treasure before the pirate does. While you will not be developing the entire game, you will write the part of the game that represents the intelligent agent, which is a pirate in this case. The pirate will try to find the optimal path to the treasure using deep Q-learning. 
# 
# You have been provided with two Python classes and this notebook to help you with this assignment. The first class, TreasureMaze.py, represents the environment, which includes a maze object defined as a matrix. The second class, GameExperience.py, stores the episodes – that is, all the states that come in between the initial state and the terminal state. This is later used by the agent for learning by experience, called "exploration". This notebook shows how to play a game. Your task is to complete the deep Q-learning implementation for which a skeleton implementation has been provided. The code blocks you will need to complete has #TODO as a header.
# 
# First, read and review the next few code and instruction blocks to understand the code that you have been given.

# In[5]:


from __future__ import print_function

import os, sys, time, datetime, json, random
import numpy as np

# Use TensorFlow’s Keras API (or plain `keras` if that works in your lab)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, PReLU
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Make sure the .py files are in your working directory
from TreasureMaze import TreasureMaze
from GameExperience import GameExperience


# The following code block contains an 8x8 matrix that will be used as a maze object:

# In[6]:


maze = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
])


# This helper function allows a visual representation of the maze object:

# In[7]:


def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    pirate_row, pirate_col, _ = qmaze.state
    canvas[pirate_row, pirate_col] = 0.3   # pirate cell
    canvas[nrows-1, ncols-1] = 0.9 # treasure cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


# The pirate agent can move in four directions: left, right, up, and down. 
# 
# While the agent primarily learns by experience through exploitation, often, the agent can choose to explore the environment to find previously undiscovered paths. This is called "exploration" and is defined by epsilon. This value is typically a lower value such as 0.1, which means for every ten attempts, the agent will attempt to learn by experience nine times and will randomly explore a new path one time. You are encouraged to try various values for the exploration factor and see how the algorithm performs.

# In[8]:


LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


# Exploration factor
epsilon = 0.1

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)


# The sample code block and output below show creating a maze object and performing one action (DOWN), which returns the reward. The resulting updated environment is visualized.

# In[9]:


qmaze = TreasureMaze(maze)
canvas, reward, game_over = qmaze.act(DOWN)
print("reward=", reward)
show(qmaze)


# This function simulates a full game based on the provided trained model. The other parameters include the TreasureMaze object and the starting position of the pirate.

# In[10]:


def play_game(model, qmaze, pirate_cell):
    qmaze.reset(pirate_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


# This function helps you to determine whether the pirate can win any game at all. If your maze is not well designed, the pirate may not win any game at all. In this case, your training would not yield any result. The provided maze in this notebook ensures that there is a path to win and you can run this method to check.

# In[11]:


def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True


# The code you have been given in this block will build the neural network model. Review the code and note the number of layers, as well as the activation, optimizer, and loss functions that are used to train the model.

# In[12]:


def build_model(maze):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model


# # #TODO: Complete the Q-Training Algorithm Code Block
# 
# This is your deep Q-learning implementation. The goal of your deep Q-learning implementation is to find the best possible navigation sequence that results in reaching the treasure cell while maximizing the reward. In your implementation, you need to determine the optimal number of epochs to achieve a 100% win rate.
# 
# You will need to complete the section starting with #pseudocode. The pseudocode has been included for you.

# In[13]:


def qtrain(model, maze, **opt):
    global epsilon

    # 1. Unpack hyperparams
    n_epoch       = opt.get('n_epoch',     15000)
    max_memory    = opt.get('max_memory',  1000)
    data_size     = opt.get('data_size',   50)
    gamma         = opt.get('discount',    0.95)
    min_epsilon   = 0.01
    epsilon_decay = 0.995

    # 2. Environment & replay buffer
    qmaze      = TreasureMaze(maze)
    experience = GameExperience(model, max_memory=max_memory, discount=gamma)

    win_history = []
    hsize       = qmaze.maze.size // 2

    start_time = datetime.datetime.now()

    # 3. Training loop
    for epoch in range(n_epoch):
        # 3.1 Pick a random starting free cell, reset env
        start_cell = random.choice(qmaze.free_cells)
        qmaze.reset(start_cell)
        state, _, _ = qmaze.act(None)  # just get the initial envstate
        total_reward = 0

        done = False
        steps = 0

        # 3.2 Play one episode
        while not done and steps < qmaze.maze.size * 2:
            prev_state = state

            # 3.2.1 ε-greedy action
            if random.random() < epsilon:
                action = random.choice(qmaze.valid_actions())
            else:
                q_vals = model.predict(prev_state)[0]
                action = int(np.argmax(q_vals))

            # 3.2.2 Step the environment
            state, reward, status = qmaze.act(action)
            done = (status != 'not_over')
            total_reward += reward

            # 3.2.3 Store to replay
            experience.remember([prev_state, action, reward, state, done])
            steps += 1

            # 3.2.4 Learn from a minibatch
            if len(experience.memory) >= data_size:
                inputs, targets = experience.get_data(data_size)
                history = model.fit(inputs, targets, epochs=1, verbose=0)
                loss = history.history['loss'][0]

        # 3.3 Track win/lose
        win = 1 if status == 'win' else 0
        win_history.append(win)

        # 3.4 Decay exploration
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # 3.5 Compute rolling win rate
        recent = win_history[-hsize:] if len(win_history) >= hsize else win_history
        win_rate = sum(recent) / len(recent)

        # 3.6 Logging
        elapsed = datetime.datetime.now() - start_time
        t = format_time(elapsed.total_seconds())
        print(f"Epoch {epoch+1:04d}/{n_epoch} | Loss {loss:.4f} "
              f"| Steps {steps:03d} | Wins {sum(win_history)} "
              f"| WinRate {win_rate:.3f} | ε {epsilon:.3f} | {t}")

        # 3.7 Early stopping if consistently winning
        if len(recent) == hsize and all(win_history[-hsize:]):
            if completion_check(model, qmaze):
                print(f"→ 100% win rate reached at epoch {epoch+1}")
                break

    # 4. Done
    total_time = format_time((datetime.datetime.now() - start_time).total_seconds())
    print(f"Training finished in {total_time}")
    return total_time


# ## Test Your Model
# 
# Now we will start testing the deep Q-learning implementation. To begin, select **Cell**, then **Run All** from the menu bar. This will run your notebook. As it runs, you should see output begin to appear beneath the next few cells. The code below creates an instance of TreasureMaze.

# In[14]:


qmaze = TreasureMaze(maze)
show(qmaze)


# In the next code block, you will build your model and train it using deep Q-learning. Note: This step takes several minutes to fully run.

# In[15]:


model = build_model(maze)
qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)


# This cell will check to see if the model passes the completion check. Note: This could take several minutes.

# In[ ]:


completion_check(model, qmaze)
show(qmaze)


# This cell will test your model for one game. It will start the pirate at the top-left corner and run play_game. The agent should find a path from the starting position to the target (treasure). The treasure is located in the bottom-right corner.

# In[ ]:


pirate_start = (0, 0)
play_game(model, qmaze, pirate_start)
show(qmaze)


# ## Save and Submit Your Work
# After you have finished creating the code for your notebook, save your work. Make sure that your notebook contains your name in the filename (e.g. Doe_Jane_ProjectTwo.ipynb). This will help your instructor access and grade your work easily. Download a copy of your IPYNB file and submit it to Brightspace. Refer to the Jupyter Notebook in Apporto Tutorial if you need help with these tasks.
