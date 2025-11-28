# Before access to data, need to test all pipelines on the manually created data for mouse 6. To test the full pipeline
# on this data, you really need a manifest file associated with this mouse. This script manually creates such a manifest
# file for the project TM-06-Pilot.

from pathlib import Path
from datetime import datetime

import pytz
import polars as pl
from sl_shared_assets import ProjectManifest


def make_manifest_from_structure(project_name: str, sessions: list[dict], out_dir: Path) -> Path:
    """sessions: list of dicts. Required keys per row:
      - animal: str|int              # your animal id
      - session: str                 # 'YYYY-MM-DD-hh-mm-ss' or 'YYYY-MM-DD-hh-mm-ss-ffffff'
      - type: str                    # any descriptive string (e.g., 'MESOSCOPE_EXPERIMENT', 'RUN_TRAINING')

    Optional keys (default False/'N/A'):
      - notes, complete, integrity, suite2p, behavior, video, dataset
        (booleans/ints for flags; they are stored as 0/1 UInt8)
    """
    eastern = pytz.timezone("America/New_York")
    animals_raw = [row["animal"] for row in sessions]
    all_numeric = all(str(a).isdigit() for a in animals_raw)

    cols = {
        "animal": [],
        "date": [],
        "session": [],
        "type": [],
        "notes": [],
        "complete": [],
        "integrity": [],
        "suite2p": [],
        "dataset": [],
        "behavior": [],
        "video": [],
    }

    for row in sessions:
        sess = row["session"]
        parts = [int(p) for p in sess.split("-")]
        if len(parts) not in (6, 7):
            raise ValueError(f"Session '{sess}' must have 6 or 7 numeric components separated by '-'")
        y, m, d, H, M, S = parts[:6]
        us = parts[6] if len(parts) == 7 else 0

        # The project’s generator interprets session timestamps as UTC, then shows them in ET.
        dt_et = datetime(y, m, d, H, M, S, us, tzinfo=pytz.UTC).astimezone(eastern)

        cols["animal"].append(int(row["animal"]) if all_numeric else str(row["animal"]))
        cols["date"].append(dt_et)
        cols["session"].append(sess)
        cols["type"].append(str(row.get("type", "N/A")))
        cols["notes"].append(str(row.get("notes", "N/A")))
        for key in ("complete", "integrity", "suite2p", "behavior", "video", "dataset"):
            cols[key].append(1 if row.get(key, False) else 0)

    schema = {
        "animal": pl.UInt64 if all_numeric else pl.String,
        "date": pl.Datetime,  # keeps the timezone from the values
        "session": pl.String,
        "type": pl.String,
        "notes": pl.String,
        "complete": pl.UInt8,
        "integrity": pl.UInt8,
        "suite2p": pl.UInt8,
        "dataset": pl.UInt8,
        "behavior": pl.UInt8,
        "video": pl.UInt8,
    }

    df = pl.DataFrame(cols, schema=schema, strict=False).sort(["animal", "session"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project_name}_manifest.feather"
    df.write_ipc(file=out_path, compression="lz4")
    return out_path


rows = [
    {
        "animal": 6,
        "session": "2025-06-23-13-32-06-980761",
        "type": "MESOSCOPE_EXPERIMENT",
        "complete": True,
        "integrity": True,
        "dataset": True,
    },
    {
        "animal": 6,
        "session": "2025-06-24-13-17-47-781337",
        "type": "MESOSCOPE_EXPERIMENT",
        "complete": True,
        "integrity": True,
        "dataset": True,
    },
    {
        "animal": 6,
        "session": "2025-06-25-16-35-01-581331",
        "type": "MESOSCOPE_EXPERIMENT",
        "complete": True,
        "integrity": True,
        "dataset": True,
    },
    {
        "animal": 6,
        "session": "2025-06-26-12-57-38-495382",
        "type": "MESOSCOPE_EXPERIMENT",
        "complete": True,
        "integrity": True,
        "dataset": True,
    },
    {
        "animal": 6,
        "session": "2025-06-27-12-44-58-770644",
        "type": "MESOSCOPE_EXPERIMENT",
        "complete": True,
        "integrity": True,
        "dataset": True,
    },
]

manifest_path = make_manifest_from_structure(
    "TM_06_pilot", rows, Path(r"C:\Users\jacob\OneDrive\Desktop\PlaceFields\slf_data")
)
pm = ProjectManifest(manifest_path)
