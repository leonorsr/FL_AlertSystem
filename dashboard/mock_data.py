"""Local fall-detection data used while the backend API is under development."""

from datetime import datetime, timedelta
from random import Random
from typing import Any, Dict, List


DATASETS = ("KFall", "SisFall", "UP-Fall")
ALERT_STATUS = ("Awaiting response", "Confirmed", "Response activated", "False positive")

PEOPLE = (
    {
        "client_id": "KF-S01",
        "dataset": "KFall",
        "person": "Ana Martins",
        "caregiver": "Carla Martins",
        "caregiver_phone": "+351 910 000 101",
        "relationship": "Mother",
        "consent": True,
        "device_status": "Protection active",
    },
    {
        "client_id": "KF-S03",
        "dataset": "KFall",
        "person": "Joaquim Silva",
        "caregiver": "Carla Martins",
        "caregiver_phone": "+351 910 000 101",
        "relationship": "Grandfather",
        "consent": True,
        "device_status": "Protection active",
    },
    {
        "client_id": "SF-SA02",
        "dataset": "SisFall",
        "person": "Maria Costa",
        "caregiver": "Inês Costa",
        "caregiver_phone": "+351 910 000 103",
        "relationship": "Mother",
        "consent": True,
        "device_status": "Protection active",
    },
    {
        "client_id": "SF-SE04",
        "dataset": "SisFall",
        "person": "António Sousa",
        "caregiver": "Paula Sousa",
        "caregiver_phone": "+351 910 000 104",
        "relationship": "Father",
        "consent": True,
        "device_status": "Protection active",
    },
    {
        "client_id": "UF-S01",
        "dataset": "UP-Fall",
        "person": "Teresa Santos",
        "caregiver": "Miguel Santos",
        "caregiver_phone": "+351 910 000 105",
        "relationship": "Mother",
        "consent": True,
        "device_status": "Protection active",
    },
    {
        "client_id": "UF-S05",
        "dataset": "UP-Fall",
        "person": "Manuel Rocha",
        "caregiver": "Sofia Rocha",
        "caregiver_phone": "+351 910 000 106",
        "relationship": "Father",
        "consent": False,
        "device_status": "Consent missing",
    },
)


def build_people() -> List[Dict[str, Any]]:
    """Return fresh person records for session-state updates."""
    return [dict(person) for person in PEOPLE]


def build_alerts(seed: int = 7) -> List[Dict[str, Any]]:
    """Return deterministic fall alerts so the UI can be developed offline."""
    rng = Random(seed)
    now = datetime.now().replace(second=0, microsecond=0)
    statuses = (
        "Awaiting response",
        "Awaiting response",
        "Awaiting response",
        "Confirmed",
        "Response activated",
        "False positive",
        "Confirmed",
        "False positive",
        "Confirmed",
        "Confirmed",
        "Response activated",
        "False positive",
    )
    alerts: List[Dict[str, Any]] = []

    for index in range(12):
        person = PEOPLE[index % len(PEOPLE)]
        status = statuses[index % len(statuses)]
        detected_at = now - timedelta(minutes=4 + index * 43 + rng.randint(0, 14))
        alerts.append(
            {
                "id": "FALL-{0:04d}".format(2051 - index),
                "detected_at": detected_at,
                "client_id": person["client_id"],
                "dataset": person["dataset"],
                "person": person["person"],
                "caregiver": person["caregiver"],
                "caregiver_phone": person["caregiver_phone"],
                "confidence": round(rng.uniform(0.86, 0.99), 2),
                "status": status,
                "source": "Automatic detection",
            }
        )

    return alerts


def build_sensor_window(client_id: str, possible_fall: bool) -> List[Dict[str, Any]]:
    """Return a small simulated sensor window until real local data is available."""
    rng = Random(sum(ord(character) for character in client_id))
    samples: List[Dict[str, Any]] = []

    for second in range(-59, 1):
        fall_zone = possible_fall and -17 <= second <= -10
        acceleration = rng.uniform(0.92, 1.08)
        rotation = rng.uniform(2.0, 8.0)
        if fall_zone:
            acceleration = rng.uniform(1.8, 3.1)
            rotation = rng.uniform(28.0, 58.0)
        samples.append(
            {
                "second": second,
                "second_end": second + 1,
                "acceleration": round(acceleration, 2),
                "rotation": round(rotation, 2),
                "zone": "Possible fall" if fall_zone else "Normal",
            }
        )

    return samples
