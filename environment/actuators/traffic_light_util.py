import traci


def determine_phases_set(sumo_phases: list[traci.trafficlight.Phase], yellow_transition_duration: float):
    # SUMO traffic signal state definitions:
    # https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#signal_state_definitions
    def _filter_phase(state: str): return 'y' not in state and (state.count('r') + state.count('s') != len(state))
    green_phases = [traci.trafficlight.Phase(duration=60, state=p.state) for p in sumo_phases if _filter_phase(p.state)]

    # Determine intermediate yellow phases which transition between distinct green phases
    yellow_transition_phases: dict[tuple[str, str], traci.trafficlight.Phase] = {}
    for _from_state in [p.state for p in green_phases]:
        for _to_state in [p.state for p in green_phases]:
            if _from_state == _to_state:
                continue
            state_str = _determine_yellow_transition(from_state=_from_state, to_state=_to_state)
            yellow_transition_phases[(_from_state, _to_state)] = \
                traci.trafficlight.Phase(duration=yellow_transition_duration, state=state_str)

    return green_phases, yellow_transition_phases


def _becomes_yellow(_from_char: str, _to_char: str):
    """Internal helper function"""
    return _from_char.lower() == "g" and any(_to_char.lower() == c for c in ("s", "r"))


def _determine_yellow_transition(from_state: str, to_state: str) -> str:
    """Returns a yellow-state that is necessary to switch from one state to another."""
    assert len(from_state) == len(to_state)
    assert from_state != to_state, "Cannot determine yellow-state between two equal trafficlight states!"
    state_str = "".join(["y" if _becomes_yellow(_from, _to) else _from for _from, _to in zip(from_state, to_state)])
    return state_str


def traffic_light_state_to_scalars(state: str) -> list[float]:
    """Converts a traffic light state string into an equal-length list of scalars."""
    scalars = []
    for _s in state:
        # SUMO traffic light state definitions:
        # https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#signal_state_definitions
        if _s.lower() in ("g", "s", "u"):  # s = green right-turn arrow,  u = red+yellow signal
            _scalar = 0
        elif _s.lower() == "y":
            _scalar = 0.5
        else:
            _scalar = 1
        scalars += [_scalar]
    return scalars


def test_determine_yellow_transition():
    transitions = [
        # Below examples come from "arterial4x4.net.xml"
        ("GGgsrrGGgsrr", "yyysrryyysrr", "srrsrGsrrsrG"),
        ("srrsrGsrrsrG", "srrsrysrrsry", "srrGGrsrrGGr"),
        ("srrGGrsrrGGr", "srrGGrsrryyr", "srrGGGsrrsrr"),
        ("srrGGGsrrsrr", "srryyysrrsrr", "srrsrrsrrGGG"),
        ("srrsrrsrrGGG", "srrsrrsrryyy", "GGgsrrGGgsrr"),
        # Below examples come from "3x3Grid2lanes.net.xml"
        ("GGGgrrrrGGGgrrrr", "YYYYrrrrYYYYrrrr", "rrrrrrrrrrrrrrrr"),
        ("rrrrGGGgrrrrGGGg", "rrrrYYYYrrrrYYYY", "rrrrrrrrrrrrrrrr"),
        # Below examples come from "nguyentl.net.xml"
        ("GGGrrr", "yyyrrr", "rrrrrr"),
        ("rrrGGG", "rrryyy", "rrrrrr"),
    ]
    for i, (_from, _expected_yellow, _to) in enumerate(transitions):
        _actual_yellow = _determine_yellow_transition(from_state=_from, to_state=_to)
        try:
            assert _expected_yellow == _actual_yellow or _expected_yellow.lower() == _actual_yellow.lower()
        except AssertionError as e:
            raise RuntimeError(f"Error in iteration {i}") from e
