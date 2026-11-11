# ElevateIQ — Intelligent Elevator Scheduling Using Q-Learning

A Python reinforcement-learning prototype for a single elevator in a 10-floor building.

## What's improved in this version
- **Valid-action masking:** the agent only explores/selects legal up/down/stop actions at a floor. This avoids wasting training steps on impossible boundary moves.
- **Masked Bellman update:** the Q-learning target considers only legal next actions.
- **More informative evaluation:** reports completed journeys, average wait until boarding, movement, people still waiting, and the total age of the remaining waiting queue.
- **Professional Pygame dashboard:** building shaft, floor call indicators, passenger queues/destinations, onboard destinations, current observation, KPIs, and keyboard controls.
- **Six training charts:** reward, completed journeys, average wait, movement, end-of-episode queue, and invalid actions.
- **Matched-seed baseline comparison:** RL and nearest-target heuristic are evaluated against the same arrival seeds.

## Requirements
- Python 3.10+
- NumPy
- Matplotlib
- Pygame

## Windows setup
Open the extracted project folder in VS Code and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```cmd
venv\Scripts\activate.bat
```

## Run order

### 1. Train the improved agent
```bash
python train.py
```
This trains for 4,000 episodes (400 steps per episode) and saves:
- `q_table.npy`
- `training_metrics.csv`
- `learning_curves.png`

Training may take several minutes depending on the computer.

### 2. Evaluate against the baseline
```bash
python evaluate.py
```
This saves:
- `evaluation_results.csv`
- `comparison_dashboard.png`

### 3. Launch the interactive dashboard
```bash
python visualize.py
```
Controls:
- SPACE: pause/resume
- UP arrow: faster
- DOWN arrow: slower
- R: reset
- ESC: exit

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

## Metrics and careful interpretation
- **Completed journeys:** passengers who reached their destination.
- **Average wait until boarding:** arrival-to-boarding wait for passengers who boarded during the episode. It does not include people still waiting at the episode end.
- **People waiting at end:** number of passengers still in queues at termination.
- **Total age of passengers still waiting:** sum of their elapsed waiting times at termination.
- **Movement:** number of one-floor elevator moves; interpret together with completed journeys and waiting.
- **Reward:** meaningful only under this exact reward function.

The baseline is a transparent nearest-target heuristic, not an optimal controller. A fair comparison uses matched random seeds, but results still depend on random demand and training settings. Do not claim RL wins every metric unless the measured output supports it.

## Team split (2 students)
**Student 1:** environment, passenger arrivals, state/action/reward definition, interface and testing.  
**Student 2:** Q-learning, training, learning curves, evaluation and comparison.  
Both members should be able to explain the full RL loop.

## Limitations
- One elevator only
- Simplified random passenger arrival process
- No peak-hour patterns, passenger priority or accessibility constraints
- State aggregation loses exact per-floor demand details
- Tabular Q-Learning may not scale to large multi-elevator systems

## Future enhancements
Multiple elevators, time-of-day demand profiles, destination-aware group scheduling, passenger priority, energy modeling, and DQN for richer state spaces.
