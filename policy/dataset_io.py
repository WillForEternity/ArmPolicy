"""Load a LeRobot v3 dataset recorded with the overhead camera into training episodes.

For each episode: joint state/action from the parquet, box features from running the same
perception code that runs on the board over the overhead video (BoxTracker, reset per episode),
then trimmed shortly after the grasp so the policy only learns the pick.

Box features are cached next to the dataset (box_features.npz) since decoding is the slow part.
"""

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

from .handover import trim_at_grasp
from .perception import BoxTracker, TableFrame

CALIB = Path(__file__).resolve().parent.parent / "table_calib.json"


def _frames(video: Path, start_s: float, end_s: float):
    """Yield BGR frames with start_s <= pts < end_s from a (multi-episode) LeRobot video file."""
    with av.open(str(video)) as c:
        stream = c.streams.video[0]
        c.seek(int(max(0.0, start_s - 1.0) / stream.time_base), stream=stream)
        for f in c.decode(stream):
            t = float(f.pts * stream.time_base)
            if t < start_s - 1e-3:
                continue
            if t >= end_s - 1e-3:
                return
            yield f.to_ndarray(format="bgr24")


def _box_features(root: Path, ep: pd.Series, video_key: str, n_frames: int) -> np.ndarray:
    v = f"videos/{video_key}/"
    path = root / "videos" / video_key / f"chunk-{int(ep[v + 'chunk_index']):03d}/file-{int(ep[v + 'file_index']):03d}.mp4"
    tracker, table, feats = BoxTracker(), None, []
    for frame in _frames(path, float(ep[v + "from_timestamp"]), float(ep[v + "to_timestamp"])):
        table = table or TableFrame((frame.shape[1], frame.shape[0]), CALIB)
        feats.append(table.features(tracker.update(frame)))
    feats = np.array(feats[:n_frames], np.float32)
    if len(feats) < n_frames:  # pad with the last pose if the video came up a frame short
        feats = np.vstack([feats, np.repeat(feats[-1:], n_frames - len(feats), 0)])
    return feats


def load_lerobot_episodes(root: Path, video_key: str | None = None, trim=True) -> list[dict]:
    info = json.loads((root / "meta/info.json").read_text())
    video_key = video_key or next(k for k, f in info["features"].items() if f["dtype"] == "video")
    meta = pd.concat(pd.read_parquet(p) for p in sorted((root / "meta/episodes").glob("*/*.parquet")))
    data = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet")))

    cache = root / "box_features.npz"
    cached = dict(np.load(cache)) if cache.exists() else {}
    episodes = []
    for _, ep in meta.sort_values("episode_index").iterrows():
        idx = int(ep["episode_index"])
        rows = data[data["episode_index"] == idx].sort_values("frame_index")
        state = np.stack(rows["observation.state"]).astype(np.float32)
        action = np.stack(rows["action"]).astype(np.float32)
        key = str(idx)
        if key not in cached:
            cached[key] = _box_features(root, ep, video_key, len(rows))
            print(f"episode {idx}: {len(rows)} frames, box found in {np.any(cached[key] != 0, 1).mean():.0%}")
        e = {"state": state, "action": action, "box": cached[key]}
        episodes.append(trim_at_grasp(e) if trim else e)
    np.savez(cache, **cached)
    return episodes
