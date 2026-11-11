def baseline_action(env):
    """
    Simple conventional baseline:
    - If a passenger onboard has a destination, target the nearest such destination.
    - Otherwise target the nearest floor with waiting passengers.
    - Stop when the elevator is at a target floor.
    This is a transparent heuristic, not a learned policy.
    """
    targets = [p["destination"] for p in env.onboard]
    if not targets:
        targets = [
            floor for floor, queue in enumerate(env.waiting) if queue
        ]

    if not targets:
        return 2  # idle/open at current floor

    # Prefer the closest target; deterministic tie-break by lower floor.
    target = min(targets, key=lambda f: (abs(f - env.floor), f))

    if target == env.floor:
        return 2
    return 0 if target > env.floor else 1
