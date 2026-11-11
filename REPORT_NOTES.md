# Report and Viva Notes — ElevateIQ

## Problem
Elevator requests arrive dynamically across floors. A fixed or simple scheduling strategy can lead to passenger delays and unnecessary movement. This prototype investigates whether a tabular Q-Learning controller can learn a useful dispatch policy in a simulated 10-floor building.

## Objective
Train an agent to choose among moving up, moving down, and opening doors/serving passengers. The reward encourages boarding and completed journeys while penalizing waiting demand, carrying load over time, and movement.

## RL formulation
- Agent: elevator controller
- Environment: one elevator, 10 floors, capacity 6
- State: floor, direction, bucketed waiting demand above/below/current floor, bucketed onboard destination directions, drop-offs at current floor, and onboard count
- Actions: UP, DOWN, STOP/OPEN DOORS
- Algorithm: tabular Q-Learning with epsilon-greedy exploration and legal-action masking

## Why action masking?
Moving up at the top floor or down at the ground floor is physically impossible. Legal-action masking excludes those choices during exploration and greedy action selection. The Bellman target also maximizes only over legal next actions. This reduces wasted exploration and prevents invalid actions from influencing the learned target.

## Training settings
- 4,000 episodes by default
- 400 steps per episode
- Learning rate: 0.10
- Discount factor: 0.95
- Initial epsilon: 1.0
- Minimum epsilon: 0.04
- Epsilon decay per episode: 0.9985

## Reward design
- +1 per passenger boarding
- +5 per completed passenger journey
- -0.10 per waiting passenger per time step
- -0.02 per onboard passenger per time step
- -0.05 per floor movement
- -2 for an invalid move as a defensive penalty

## Evaluation protocol
The trained greedy policy and a nearest-target heuristic baseline are evaluated over 100 matched random seeds. Both policies experience the same seeded passenger-arrival stream for each paired episode.

Report these metrics:
1. Completed passenger journeys
2. Average wait from arrival until boarding
3. Floor movements
4. Number of people still waiting at episode end
5. Total elapsed waiting age of people still waiting at episode end
6. Cumulative reward

Important: average wait until boarding excludes people who remain unboarded at termination. Therefore, interpret it together with the end-of-episode queue metrics. The nearest-target baseline is a heuristic, not an optimal scheduler.

## Viva questions
**Why Q-Learning?** It is model-free and straightforward for a compact discrete state/action problem.

**What is the Q-table?** A table of estimated long-term returns for each represented state and action.

**What is epsilon-greedy?** A policy that explores randomly with probability epsilon and otherwise chooses the highest-valued legal action.

**Why mask actions?** To prevent physically impossible moves from being selected or included in the next-state maximum.

**How do you show learning?** Plot episode reward, completed journeys, waiting time and queue-at-end metrics across training, then evaluate the greedy learned policy against a baseline on matched demand seeds.

**Does a higher reward prove better real-world service?** No. Reward depends on chosen weights. Service metrics must also be compared.

## Limitations
- Single elevator and simplified passenger demand
- No time-of-day demand profiles or passenger priority
- State bucketing discards exact queue/destination details
- The baseline is a simple nearest-target heuristic
- Results depend on random seeds, reward design and training duration

## Future work
Multiple elevators, peak-hour arrivals, priority passengers, energy consumption, richer state representations and DQN.

## Reference evaluation using the bundled prior trained Q-table
The included Q-table is compatible with the current state dimensions and is supplied so the interface can be demoed immediately. The current `train.py` retrains it using legal-action masking. Always run `python evaluate.py` after your final training and use the resulting CSV values in the submitted report; do not reuse old metrics as if they came from the new training run.
