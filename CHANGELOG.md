# Version 2 improvements

1. Added legal-action masking to epsilon-greedy action selection.
2. Added legal-action masking to the Q-learning Bellman target.
3. Added explicit metrics for passenger arrivals, completed journeys, end queues, and elapsed waiting age of passengers still waiting.
4. Expanded training plots to six panels, including end-of-episode queue and invalid-action count.
5. Added matched-seed baseline evaluation and a comparison dashboard.
6. Rebuilt the Pygame interface with a dark operations-center layout, floor call indicators, passenger queue destinations, onboard destinations, observation details, KPI cards and keyboard controls.

The bundled Q-table and files prefixed `previous_` are from the earlier 4,000-episode run. They are included to let the demo launch immediately. Run `python train.py` to create a fresh Q-table and fresh training metrics for this improved version, then run `python evaluate.py` again before recording final results.
