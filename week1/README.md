# Week 1: Introduction to Reinforcement Learning & Multi-Armed Bandits

**Course:** Reinforcement Learning - Continuing Education  
**Institution:** Zurich University of Applied Sciences

---

## 🎯 Learning Objectives

By the end of this practical lab exercise, you will be able to:
- Understand the basic RL paradigm: agent, environment, actions, and rewards
- Explain the exploration vs exploitation tradeoff
- Implement epsilon-greedy and UCB algorithms from scratch
- Compare and analyze different exploration strategies
- Visualize and interpret RL algorithm performance

## 📁 Materials

### Lecture Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/week1/week1_lecture_examples.ipynb)

Interactive notebook used during lecture to demonstrate RL concepts.

### Lab Assignment
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/week1/week1_lab_assignment.ipynb)

Your hands-on exercise - implement bandit algorithms from scratch and analyze their performance.

### Solutions
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/week1/week1_lab_solutions.ipynb)

Complete implementations with detailed explanations. **Available after the submission deadline.**

## 🚀 Getting Started

### Using Google Colab
1. Click the "Open in Colab" button above
2. The notebook will open in Google Colab
3. Go to **Runtime → Run all** or execute cells individually
4. All required libraries (numpy, matplotlib) are pre-installed in Colab

### Running Locally (Optional)
If you prefer to run locally:
```bash
pip install numpy matplotlib jupyter
jupyter notebook week1_lab_assignment.ipynb
```

## 📊 Lab Structure

Your lab assignment has 7 parts:

**Part 1: Bandit Environment**  
Implement a multi-armed bandit with Gaussian reward distributions.

**Part 2: Epsilon-Greedy Agent**  
Implement epsilon-greedy action selection and incremental value updates.

**Part 3: UCB Agent**  
Implement Upper Confidence Bound algorithm with confidence bonus.

**Part 4: Experiment Runner**  
Create infrastructure to run and analyze multiple experiments.

**Part 5: Epsilon Comparison**  
Compare epsilon-greedy with different ε values (0, 0.01, 0.1, 0.3).

**Part 6: Strategy Comparison**  
Compare epsilon-greedy vs UCB performance.

**Part 7: Analysis Questions**  
Written analysis of your experimental results.

**Bonus Challenges (Optional)**  
Try decaying epsilon, optimistic initial values, or other advanced techniques.

## 🔑 Key Concepts

### Multi-Armed Bandit
The simplest RL problem where you choose between multiple slot machines ("arms"), each with different reward distributions. Your goal: maximize total rewards over many plays.

### Exploration vs Exploitation
- **Exploration:** Try different actions to learn about them
- **Exploitation:** Use current knowledge to get the best reward
- **The Challenge:** You need both, but in what balance?

### Epsilon-Greedy
A simple strategy: with probability ε (e.g., 0.1), pick a random arm to explore. Otherwise, pick the best-known arm. Easy to implement and surprisingly effective!

### Upper Confidence Bound (UCB)
A smarter strategy that automatically balances exploration and exploitation. It prioritizes arms with high estimated values OR high uncertainty (haven't tried much). Better theoretical guarantees than epsilon-greedy.

## 💡 Tips for Success

**As You Work Through the Lab:**
1. Read each TODO comment carefully before coding
2. Test each part before moving to the next
3. Run cells frequently to catch errors early
4. Compare your plots to expected behavior
5. Don't skip the analysis questions - they're crucial!

**Common Pitfalls:**
- **Array indexing:** Remember Python uses 0-based indexing (Arm 0, Arm 1, Arm 2...)
- **Incremental updates:** Use the formula: `Q[a] += (reward - Q[a]) / N[a]`
- **UCB division by zero:** Handle the case when `N[a] = 0`
- **Random vs randn:** Use `np.random.randn()` for Gaussian, not `np.random.rand()`

**Debugging Strategy:**
- Print intermediate values to understand what's happening
- Visualize Q-values and arm counts as you go
- Start with small numbers (3 arms, 100 steps) for testing
- Check that optimal arm matches `np.argmax(bandit.means)`

## 🎓 Understanding Your Results

**What Good Results Look Like:**

*Epsilon-Greedy (ε=0.1):*
- Should reach 70-85% optimal action rate
- Average reward converges close to optimal arm's mean
- Better than greedy but not perfect

*UCB (c=2.0):*
- Should reach 90-95% optimal action rate
- Faster initial learning than epsilon-greedy
- Lower cumulative regret over time

*Greedy (ε=0):*
- Often gets stuck on suboptimal arms
- Performance varies wildly between runs
- Low optimal action rate (30-50%)

## 🔗 Looking Ahead

**Next Week: Markov Decision Processes (MDPs)**  
We'll extend bandits to problems with:
- Multiple states (not just one decision)
- State transitions (actions affect future situations)
- Delayed rewards (planning ahead matters)
- Q-learning and dynamic programming

## 📚 Additional Resources

**Want to Learn More?**
- Sutton & Barto (2018): "Reinforcement Learning: An Introduction" - Chapter 2
- [Multi-Armed Bandits Overview](https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/)
- [OpenAI Spinning Up: RL Introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

**Interesting Extensions:**
- Contextual bandits (decisions depend on context)
- Thompson Sampling (Bayesian approach)
- Non-stationary bandits (reward distributions change over time)

---

**Questions?** Ask during class or office hours. Good luck and enjoy your first RL implementation! 🎰🤖
