import numpy as np


class ElevatorEnv:
    """
    A small, tabular-RL-friendly elevator simulator.

    Floors are numbered 0 .. num_floors-1.
    Each step represents one time unit.
    Actions: 0=UP, 1=DOWN, 2=STOP/OPEN DOOR.
    """

    def __init__(self, num_floors=10, capacity=6, max_steps=400,
                 arrival_probability=0.15, seed=None):
        self.num_floors = num_floors
        self.capacity = capacity
        self.max_steps = max_steps
        self.arrival_probability = arrival_probability
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.floor = 0
        self.direction = 0       # -1 down, 0 idle, +1 up
        self.onboard = []        # passenger dictionaries
        self.waiting = [[] for _ in range(self.num_floors)]
        self.time = 0
        self.total_served = 0
        self.total_wait = 0
        self.boarded_count = 0
        self.total_movement = 0
        self.total_invalid_actions = 0
        self.waiting_time_at_end = 0
        self.passenger_arrivals = 0
        self.total_completed_journeys = 0
        return self.get_state()

    def _bucket(self, value):
        # 0, 1, or 2 meaning 2 or more
        return min(int(value), 2)

    def get_state(self):
        above = sum(len(self.waiting[f]) for f in range(self.floor + 1, self.num_floors))
        below = sum(len(self.waiting[f]) for f in range(0, self.floor))
        here = len(self.waiting[self.floor])
        onboard_above = sum(p["destination"] > self.floor for p in self.onboard)
        onboard_below = sum(p["destination"] < self.floor for p in self.onboard)
        onboard_here = sum(p["destination"] == self.floor for p in self.onboard)
        return (
            int(self.floor),
            int(self.direction + 1),  # map -1,0,+1 to 0,1,2
            self._bucket(above),
            self._bucket(below),
            self._bucket(onboard_above),
            self._bucket(onboard_below),
            self._bucket(here),
            self._bucket(onboard_here),
            self._bucket(len(self.onboard)),
        )

    def valid_actions(self):
        """Return legal actions for the current floor."""
        actions = [2]  # opening doors / serving at the current floor is always legal
        if self.floor < self.num_floors - 1:
            actions.append(0)
        if self.floor > 0:
            actions.append(1)
        return actions

    def _generate_passenger(self):
        if self.rng.random() < self.arrival_probability:
            origin = int(self.rng.integers(0, self.num_floors))
            destination = int(self.rng.integers(0, self.num_floors - 1))
            if destination >= origin:
                destination += 1
            self.waiting[origin].append({
                "origin": origin,
                "destination": destination,
                "arrival_time": self.time,
            })
            self.passenger_arrivals += 1

    def _serve_floor(self):
        # First let passengers whose destination is this floor leave.
        remaining = []
        completed = 0
        for passenger in self.onboard:
            if passenger["destination"] == self.floor:
                completed += 1
            else:
                remaining.append(passenger)
        self.onboard = remaining
        self.total_served += completed
        self.total_completed_journeys += completed

        # Then board passengers waiting at this floor, up to capacity.
        space = self.capacity - len(self.onboard)
        boarded = self.waiting[self.floor][:space]
        self.waiting[self.floor] = self.waiting[self.floor][space:]
        for passenger in boarded:
            wait = self.time - passenger["arrival_time"]
            self.total_wait += wait
            self.boarded_count += 1
            self.onboard.append(passenger)
        return completed, len(boarded)

    def step(self, action):
        if action not in (0, 1, 2):
            raise ValueError("Action must be 0 (UP), 1 (DOWN), or 2 (STOP).")

        self._generate_passenger()
        reward = 0.0
        completed = 0
        boarded = 0
        moved = False

        if action == 0:  # UP
            if self.floor < self.num_floors - 1:
                self.floor += 1
                self.direction = 1
                moved = True
            else:
                reward -= 2.0
                self.total_invalid_actions += 1
        elif action == 1:  # DOWN
            if self.floor > 0:
                self.floor -= 1
                self.direction = -1
                moved = True
            else:
                reward -= 2.0
                self.total_invalid_actions += 1
        else:  # STOP / open doors
            completed, boarded = self._serve_floor()
            self.direction = 0
            reward += 1.0 * boarded

        if moved:
            self.total_movement += 1
            reward -= 0.05

        # Shaped feedback: boarding and completing trips are useful;
        # delays and carrying passengers for too long are costly.
        waiting_now = sum(len(queue) for queue in self.waiting)
        # Small per-step delay costs encourage prompt service without overwhelming
        # the positive feedback for boarding and completing journeys.
        reward -= 0.10 * waiting_now
        reward -= 0.02 * len(self.onboard)
        reward += 5.0 * completed

        self.time += 1
        done = self.time >= self.max_steps
        if done:
            self.waiting_time_at_end = sum(
                self.time - passenger["arrival_time"]
                for queue in self.waiting for passenger in queue
            )
        info = {
            "completed_this_step": completed,
            "boarded_this_step": boarded,
            "waiting": waiting_now,
            "onboard": len(self.onboard),
            "served": self.total_served,
            "average_wait": (
                self.total_wait / self.boarded_count if self.boarded_count else 0.0
            ),
            "movement": self.total_movement,
            "invalid_actions": self.total_invalid_actions,
            "arrivals": self.passenger_arrivals,
            "completed_journeys": self.total_completed_journeys,
        }
        return self.get_state(), reward, done, info

    def metrics(self):
        return {
            "served": self.total_served,
            "average_wait": (
                self.total_wait / self.boarded_count if self.boarded_count else 0.0
            ),
            "movement": self.total_movement,
            "invalid_actions": self.total_invalid_actions,
            "waiting_at_end": sum(len(q) for q in self.waiting),
            "onboard_at_end": len(self.onboard),
            "arrivals": self.passenger_arrivals,
            "completed_journeys": self.total_completed_journeys,
            "waiting_time_at_end": self.waiting_time_at_end,
            "total_waiting_people": sum(len(q) for q in self.waiting),
        }
