# Week 2: Q-Learning


## Learning Objectives

By the end of this session, you will be able to:
- Understand the difference between value-based and policy-based learning
- Explain the Q-learning update rule and why it works
- Implement Q-learning for turn-based games
- Train agents and play against them interactively
- Analyze how state space complexity affects learning

## Materials

### Lecture: Q-Learning with Tic-Tac-Toe
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhaw-physical-ai/adml-rl/blob/main/week2/week2_lecture_tictactoe.ipynb)

Follow-along notebook demonstrating Q-learning concepts step by step:
- From bandits to MDPs
- Value functions and the Bellman equation
- Building a Tic-Tac-Toe agent from scratch
- Watching the agent improve over time
- Visualizing learned Q-values

### Lab Assignment: Q-Learning for Nim
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhaw-physical-ai/adml-rl/blob/main/week2/week2_lab_nim.ipynb)

Hands-on exercise where you:
- Implement the Q-learning update rule
- Train agents on progressively harder Nim configurations
- Play interactively against your trained agents
- **Challenge:** Can you beat the agent at 4 piles?

### Solutions
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhaw-physical-ai/adml-rl/blob/main/week2/week2_lab_nim_solutions.ipynb)

Complete solutions with detailed explanations. **Available after the submission deadline.**

## 🎮 The Game of Nim

Nim is a mathematical strategy game:
- One or more **piles** of objects
- Two players take turns
- On your turn: remove **any number** (≥1) from **one pile**
- **The player who takes the last object LOSES**

### Why Nim?
- Perfect for tabular Q-learning
- Trains in seconds
- Natural difficulty progression (1 pile → 4 piles)
- Mathematical optimal strategy exists (but it's not obvious!)

## 🚀 Getting Started

### Using Google Colab (Recommended)
1. Click the "Open in Colab" button above
2. All libraries are pre-installed
3. Run cells with Shift+Enter

### Running Locally
```bash
pip install numpy matplotlib jupyter
jupyter notebook week2_lab_nim.ipynb
```

## Lab Structure

### Part 1-2: Understand the Environment
Run cells to see how Nim works. Simple variable changes only.

### Part 3: Implement Q-Learning Update ⭐
**The key task:** Complete one line of code
```python
self.Q[(state, action)] = current_q + self.alpha * (reward - current_q)
```

### Part 4-6: Train and Play
- Train on 1, 2, 3, and 4 piles
- Play interactively against your agents
- See if you can beat them!

### Part 7: Evaluation
Compare agent performance across configurations.

### Part 8: Analysis Questions
Reflect on what you observed.

### Bonus Challenges (Optional)
- Self-play training
- Learning curve visualization
- Hyperparameter exploration

## Key Concepts

### From Bandits to MDPs
| Multi-Armed Bandits | Markov Decision Processes |
|---------------------|---------------------------|
| Single state | Multiple states |
| Immediate rewards | Delayed rewards |
| Find best action | Find best action **per state** |

### The Q-Learning Update
```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```
- **α (alpha):** Learning rate - how fast we update
- **γ (gamma):** Discount factor - importance of future rewards
- **r:** Immediate reward
- **max Q(s',a'):** Best expected future value

### Value vs Policy Learning
| Value-Based (Q-Learning) | Policy-Based |
|--------------------------|--------------|
| Learn Q(s,a) values | Learn π(a\|s) directly |
| Policy: pick argmax Q | Explicit policy function |
| Good for discrete actions | Good for continuous actions |

## Tips for Success

### The One Line You Need to Implement
```python
# In the learn() method:
self.Q[(state, action)] = current_q + self.alpha * (reward - current_q)
```

This moves the Q-value toward the observed reward. That's it!

### Playing Against the Agent
1. Uncomment the `play_against_agent(...)` line
2. Run the cell
3. Enter moves as: `pile, amount` (e.g., `0, 2` removes 2 from pile 0)

### Winning at Nim
- **1 pile:** Leave opponent with exactly 1 object
- **2 piles:** Make the piles equal, then mirror opponent's moves
- **3+ piles:** The XOR strategy (ask about this in class!)

### Common Issues
- If Q-values all stay 0: Check your update formula
- If agent plays randomly: Make sure epsilon=0 during evaluation
- If training is slow: You may need more games for complex configurations

## Challenge Progression

| Config | Difficulty | Can You Win? |
|--------|------------|--------------|
| [5] | Easy | Yes, with the right first move |
| [3, 4] | Medium | Yes, make piles equal |
| [2, 3, 4] | Hard | Requires deeper strategy |
| [1, 3, 5, 7] | Expert | Only if you go second! |

## Connection to Next Week

This week: **Tabular Q-Learning**
- Works great for small state spaces
- Q-table stores one value per (state, action) pair
- Limited by memory and exploration

Next week: **Deep Q-Networks (DQN)**
- Neural network replaces Q-table
- Generalizes to unseen states
- Scales to Atari games and beyond!

## Additional Resources

**Theory:**
- "RL: An Introduction" - Chapter 6 (TD Learning)
- [Q-Learning Explained](https://www.freecodecamp.org/news/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc/)

**Nim Strategy:**
- [Nim Game Theory](https://en.wikipedia.org/wiki/Nim#Mathematical_theory)
- The XOR trick: Position is losing if pile1 ⊕ pile2 ⊕ ... = 0

**Fun Extensions:**
- [OpenAI Gym Environments](https://gymnasium.farama.org/)
- [AlphaGo Documentary](https://www.youtube.com/watch?v=WXuK6gekU1Y) - Self-play at scale!

