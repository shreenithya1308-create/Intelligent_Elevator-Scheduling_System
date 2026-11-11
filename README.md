# ElevateIQ 🚀
### Intelligent Elevator Scheduling Using Reinforcement Learning

An AI-powered elevator scheduling simulation that uses **Q-Learning** to learn intelligent elevator movement and passenger scheduling decisions.

## ✨ Features

- 🧠 Q-Learning based intelligent scheduling
- 🏢 10-floor elevator simulation
- 👥 Dynamic passenger request generation
- 🎯 Reward-based learning and decision-making
- 📊 Training performance and learning curves
- 📈 Q-Learning vs heuristic baseline comparison
- 🖥️ Interactive Pygame dashboard
- 📉 Passenger waiting time and elevator movement analysis

## 🖥️ Application Preview

**ElevateIQ – Intelligent Elevator Control Dashboard**

![ElevateIQ Dashboard](screenshots/elevator_interface.png)

## 🛠️ Built With

- Python
- Q-Learning
- NumPy
- Pandas
- Matplotlib
- Pygame

## 📂 Project Structure

```text
Intelligent-Elevator-RL/
│
├── src/              # Core Python implementation
├── models/           # Trained Q-Learning model
├── results/          # Training graphs and metrics
├── screenshots/      # Application screenshots
│
├── README.md
├── requirements.txt
├── evaluation_results.csv
├── CHANGELOG.md
└── .gitignore
```

## RL formulation

**Agent:** elevator controller  
**Environment:** simulated 10-floor building  
**State:** `(floor, direction, waiting_above, waiting_below, onboard_destinations_above, onboard_destinations_below, waiting_here, dropoffs_here, onboard_count)`; passenger counts are bucketed as 0, 1, or 2+  
**Actions:** `0=UP`, `1=DOWN`, `2=STOP/OPEN DOORS`  
**Policy:** epsilon-greedy  
**Algorithm:** tabular Q-Learning with legal-action masking

Q-learning update:

`Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]`

At terminal steps, the future-value term is omitted. At non-terminal steps, the maximum is taken only over legal actions in the next state.

## Reward design
- +1 per passenger boarding
- +5 per completed passenger journey
- -0.10 per waiting passenger per time step
- -0.02 per onboard passenger per time step
- -0.05 per floor movement
- -2 for an invalid move (defensive fallback; masked training/evaluation should not choose one)

Reward weights are experimental design choices. They should be reported and not described as universally optimal.

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Intelligent-Elevator-RL
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

Run the following commands from the project root:

**Train the agent**

```bash
python src/train.py
```

**Evaluate the agent**

```bash
python src/evaluate.py
```

**Launch the visualization**

```bash
python src/visualize.py
```

Run training first if a trained Q-table is not already available.

Controls:
- SPACE: pause/resume
- UP arrow: faster
- DOWN arrow: slower
- R: reset
- ESC: exit

## 📊 Performance Analysis

The system evaluates the learned policy using:

- Cumulative reward
- Completed passenger journeys
- Average passenger waiting time
- Elevator floor movements
- Remaining waiting passengers

The learned Q-Learning policy is compared with a nearest-target heuristic baseline.

Training curves and comparison results are available in the `results/` directory.

## 🔮 Future Enhancements

- Multi-elevator scheduling
- Deep Q-Network (DQN)
- Multi-agent Reinforcement Learning
- Realistic passenger arrival patterns
- Advanced elevator traffic optimization

## 👩‍💻 Project Team

**Team Size:** 2 Students

**Domain:** Artificial Intelligence | Reinforcement Learning

---

⭐ *ElevateIQ – Learning Smarter Elevator Scheduling Through Reinforcement Learning.*
