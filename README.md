# cs370-pirate-intelligent-agent
What this project is
A pathfinding agent (“pirate”) trained with deep Q-learning to reach the treasure in an 8×8 maze. The notebook implements the training loop, evaluation helpers, and logging; the provided .py files define the environment and replay buffer.

What code was given vs. what I wrote

Given: TreasureMaze.py (environment, reward shaping, valid action logic) and GameExperience.py (experience replay and Bellman targets).

Created by me (in the notebook):

The DQN model (two dense layers with PReLU; 4-unit output for actions).

The full training loop (qtrain): ε-greedy policy, action masking to avoid illegal moves, warm-up before fitting, interval training, early-stopping check, and clear per-epoch logging.

Helpers (play_game, completion_check, show) usage to evaluate wins and visualize the path.

Small curriculum: start at (0,0) first, then randomize starting cells.

CPU-only setup for Apporto/Colab, plus optional checkpointing.

I built a pirate “bot” to find the fastest valid path to the treasure on an 8×8 grid. I used deep Q-learning with a small feed-forward network to score actions. A maze environment (cells, rewards, legal moves), a memory of past steps, and a DQN training loop that balances trying new moves with sticking to what works.  From the start I kept an eye on ethics and rigor, making sure results were reproducible, resources were used sensibly, and the reward design didn’t let the agent “cheat” its way to a high score without actually reaching the goal.

This project reminded me what computer scientists do at a practical level: take a messy goal, model it clearly, pick algorithms that can actually deliver, and ship something dependable. Turning “find the treasure fast” into a working agent required that kind of translation from idea to system, with tests and metrics that tell the truth about progress.

My process now is more disciplined. I begin with a small, testable slice: a clean environment, clear rewards, and a baseline agent. Then I add guardrails fixed seeds, checkpoints, and simple logging so I can see what’s happening and repeat it. For training, I used action masking to prevent illegal moves, a short warm-up before fitting, interval updates to avoid thrashing, and a light curriculum that starts at (0,0) before randomizing starts.

On the ethics side, I focused on honesty and efficiency. I locked things down with fixed random seeds, regular checkpoints, and versioned code/data so runs are repeatable. Illegal moves and revisits get dinged, and the only positive payout comes when the treasure is actually reached, so there’s no easy way for the agent to game the task. I also documented assumptions and limits, like how exploration settings and reward shaping affect results.

Course topics showed up everywhere in the build. ε-greedy trade-offs, Bellman targets, and replay memory were not just terms from slides they solved concrete problems like instability and slow convergence. Good engineering habits modular code, clear logs, and reproducible runs made the experiments manageable and the outcomes trustworthy.
