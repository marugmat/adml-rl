# Week 3: Policy Gradient Methods - Local Setup Guide

This guide helps you run the REINFORCE training script on your personal laptop (Windows, Mac, or Linux).

## Objectives

In this lab, you will:
1. Understand the policy network architecture
2. Implement action selection from a stochastic policy
3. Compute discounted returns
4. Implement the REINFORCE policy gradient loss
5. (Optional) Add a baseline for variance reduction

### Lab Assignment: CartPole
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhaw-physical-ai/adml-rl/blob/main/week3/week3_lab_assignment.ipynb)

### Lab Assignment: Solutions
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhaw-physical-ai/adml-rl/blob/main/week3/week3_lab_solutions.ipynb)

### Supporting Notebook: Policy gradient
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhaw-physical-ai/adml-rl/blob/main/week3/week3_policy_gradient.ipynb)

## The CartPole Task

**Goal:** Balance a pole on a moving cart for as long as possible

**State (4 values):**
- Cart position
- Cart velocity
- Pole angle
- Pole angular velocity

**Actions (2 discrete):**
- 0: Push left
- 1: Push right

**Reward:** +1 for each timestep the pole stays up

**Success:** Average reward ≥ 475 over 100 episodes

---

# Local Cartpole Example

## Prerequisites

- Python 3.8 or higher
- `conda` or `pip` for package management
- 2GB free disk space
- (Optional) GPU for faster training

---

## Setup Instructions

### Option 1: Using Conda (Recommended)

Conda works on Windows, Mac, and Linux and manages dependencies cleanly.

#### Step 1: Install Miniconda

If you don't have conda installed:

- **Windows/Mac/Linux**: Download from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- Follow the installer instructions for your OS

#### Step 2: Create Environment

```bash
# Create a new environment named 'rl_course'
conda create -n rl_course python=3.10 -y

# Activate the environment
conda activate rl_course
```

#### Step 3: Install Dependencies

```bash
# Install PyTorch (CPU version)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install other packages
conda install numpy matplotlib -y
pip install gymnasium[classic-control]
```

**For GPU support** (if you have NVIDIA GPU):
```bash
# Instead of cpuonly, use:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

---

### Option 2: Using pip

If you prefer pip over conda:

#### Step 1: Create Virtual Environment

**On Linux/Mac:**
```bash
python3 -m venv rl_env
source rl_env/bin/activate
```

**On Windows:**
```cmd
python -m venv rl_env
rl_env\Scripts\activate
```

#### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium[classic-control] numpy matplotlib
```

---

## Running the Training Script

### Basic Training

```bash
python train_cartpole.py
```

This will train for 1000 episodes with default hyperparameters.

### Custom Hyperparameters

```bash
# Train for 2000 episodes with learning rate 0.005
python train_cartpole.py --episodes 2000 --lr 0.005

# Use different gamma (discount factor)
python train_cartpole.py --gamma 0.95

# Larger network
python train_cartpole.py --hidden-dim 64

# Custom seed for reproducibility
python train_cartpole.py --seed 123
```

### All Available Options

```bash
python train_cartpole.py --help
```

**Options:**
- `--episodes`: Number of training episodes (default: 1000)
- `--lr`: Learning rate (default: 0.01)
- `--gamma`: Discount factor (default: 0.99)
- `--hidden-dim`: Hidden layer size (default: 32)
- `--seed`: Random seed (default: 42)
- `--save-model`: Path to save model (default: trained_policy.pth)
- `--no-plot`: Disable plotting (useful for servers)

---

## Visualizing Trained Agent

After training, visualize your agent:

```bash
python train_cartpole.py --visualize --model trained_policy.pth
```

This opens a window showing your trained agent balancing the pole!

**Note:** On remote servers without display, use `--no-plot` during training.

---

## Output Files

After training, you'll find:

1. **`trained_policy.pth`** - Saved model weights
2. **`training_curve.png`** - Learning curve plot

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'gymnasium'`

**Solution:**
```bash
pip install gymnasium[classic-control]
```

### Issue: Rendering doesn't work on Linux

**Solution:** Install display dependencies:
```bash
sudo apt-get install python3-opengl xvfb
# Run with virtual display
xvfb-run -a python train_cartpole.py --visualize
```

### Issue: Training is very slow

**Solutions:**
1. Reduce episodes: `--episodes 500`
2. Use GPU if available (see conda GPU instructions)
3. Smaller network: `--hidden-dim 16`

### Issue: Agent doesn't learn well

**Solutions:**
1. Train longer: `--episodes 2000`
2. Adjust learning rate: `--lr 0.005` or `--lr 0.02`
3. Try different seed: `--seed 999`
4. Check your implementations in the lab notebook!

### Issue: `ImportError` or `DLL load failed` on Windows

**Solution:**
- Make sure you're using Python 3.8-3.11 (not 3.12+)
- Reinstall PyTorch:
  ```bash
  pip uninstall torch
  pip install torch torchvision torchaudio
  ```

---

## Tips for Experimentation

### Good Hyperparameters to Try

```bash
# Conservative (more stable)
python train_cartpole.py --lr 0.005 --gamma 0.99

# Aggressive (faster but less stable)
python train_cartpole.py --lr 0.02 --gamma 0.95

# Longer training with patience
python train_cartpole.py --episodes 2000 --lr 0.01
```

### Comparing Different Runs

```bash
# Run multiple experiments
python train_cartpole.py --seed 1 --save-model model_seed1.pth
python train_cartpole.py --seed 2 --save-model model_seed2.pth
python train_cartpole.py --seed 3 --save-model model_seed3.pth

# Compare their performance
python train_cartpole.py --visualize --model model_seed1.pth
python train_cartpole.py --visualize --model model_seed2.pth
```

---

## Understanding the Output

### During Training

```
Episode  100 | Avg Reward:  45.23 | Last Episode:  67
Episode  200 | Avg Reward: 123.45 | Last Episode: 156
  → Saved new best model (avg: 123.45)
Episode  300 | Avg Reward: 234.56 | Last Episode: 241
...
```

- **Avg Reward**: Rolling average over last 100 episodes
- **Last Episode**: Reward from most recent episode
- Model is saved when average improves

### Success Criteria

- **Target**: Average reward ≥ 475 over 100 episodes
- Typical training time: 500-1000 episodes
- CartPole is "solved" when consistently hitting 475+

---

## Platform-Specific Notes

### Windows

- Use Command Prompt or PowerShell
- Activate environment: `rl_env\Scripts\activate`
- If you see SSL errors, update pip: `python -m pip install --upgrade pip`

### Mac (Apple Silicon M1/M2/M3)

```bash
# Use ARM-optimized PyTorch
conda install pytorch::pytorch -c pytorch
```

### Linux

- Most straightforward platform
- Use system package manager for system dependencies
- For headless servers, use `--no-plot` and copy PNG files

---

## Additional Resources

### Official Documentation

- Gymnasium: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- PyTorch: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)

### Experiment Ideas

1. **Different Environments**: Try `MountainCar-v0`, `Acrobot-v1`, `LunarLander-v2`
   ```bash
   # Modify train_cartpole.py to use different env
   env = gym.make('MountainCar-v0')
   ```

2. **Hyperparameter Search**: Write a script to try multiple learning rates

3. **Baseline Comparison**: Implement the value baseline from Task 5

---

## Getting Help

If you encounter issues:

1. Check error message carefully
2. Verify all packages installed: `pip list | grep torch`
3. Try in Google Colab first to isolate environment issues
4. Ask during lab session

**Common Success Checklist:**
- ✅ Python 3.8-3.11 installed
- ✅ Virtual environment activated
- ✅ All packages installed without errors
- ✅ Script runs without import errors
- ✅ Training shows progress (rewards increasing)

---

## Next Steps

Once you're comfortable with the standalone script:

1. Complete the optional Task 5 (value baseline)
2. Experiment with different hyperparameters
3. Try other Gymnasium environments
4. Prepare for Week 4: Actor-Critic methods!


