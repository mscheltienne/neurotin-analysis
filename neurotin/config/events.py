"""ID and duration of events present in NeuroTin recordings."""

EVENTS = {
    "rest": 1,
    "blink": 2,
    "resting-state": 3,
    "audio": 4,
    "regulation": 5,
    "non-regulation": 6,
}
EVENTS_MAPPING = {
    1: "rest",
    2: "blink",
    3: "resting-state",
    4: "audio",
    5: "regulation",
    6: "non-regulation",
}
EVENTS_DURATION_MAPPING = {
    1: 1,
    2: 60,
    3: 120,
    4: 0.8,
    5: 16,
    6: 8,
}
FIRST_REST_PHASE_EXT = 7  # extension of the first rest phase in seconds.
