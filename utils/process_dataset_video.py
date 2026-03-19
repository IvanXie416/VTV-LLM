import argparse
import json
import os
import random
from pathlib import Path

try:
    from .constants_video import TEST_OBJECTS, TRAIN_OBJECTS, VAL_OBJECTS
except ImportError:
    from constants_video import TEST_OBJECTS, TRAIN_OBJECTS, VAL_OBJECTS


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _extract_object_name(filename):
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def _append_sample(sample_map, object_name, video_path):
    sample_map.setdefault(object_name, []).append(video_path)


def get_samples(dataset_path, train_json_path, val_json_path, test_json_path, seed=0):
    rng = random.Random(seed)
    train_objects = list(TRAIN_OBJECTS)
    rng.shuffle(train_objects)

    dataset_files = sorted(os.listdir(dataset_path))
    video_files = [
        f
        for f in dataset_files
        if os.path.isfile(os.path.join(dataset_path, f)) and Path(f).suffix.lower() in VIDEO_EXTENSIONS
    ]

    train_sample_paths = {}
    val_sample_paths = {}
    test_sample_paths = {}

    for video_file in video_files:
        object_name = _extract_object_name(video_file)
        video_path = os.path.join(dataset_path, video_file)

        if len(VAL_OBJECTS) == 0:
            if object_name in train_objects:
                if rng.random() < 0.8:
                    _append_sample(train_sample_paths, object_name, video_path)
                else:
                    _append_sample(val_sample_paths, object_name, video_path)
        else:
            if object_name in train_objects:
                _append_sample(train_sample_paths, object_name, video_path)
            if object_name in VAL_OBJECTS:
                _append_sample(val_sample_paths, object_name, video_path)

        if object_name in TEST_OBJECTS:
            _append_sample(test_sample_paths, object_name, video_path)

    with open(train_json_path, "w") as f:
        json.dump(train_sample_paths, f, indent=2, sort_keys=True)
    with open(val_json_path, "w") as f:
        json.dump(val_sample_paths, f, indent=2, sort_keys=True)
    with open(test_json_path, "w") as f:
        json.dump(test_sample_paths, f, indent=2, sort_keys=True)

    print(
        "Split summary: "
        f"train={sum(len(v) for v in train_sample_paths.values())}, "
        f"val={sum(len(v) for v in val_sample_paths.values())}, "
        f"test={sum(len(v) for v in test_sample_paths.values())}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="vtvllm/vbts_video", help="video directory")
    parser.add_argument("--output_path", default="vtvllm/vbts_video", help="processed samples directory")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducible split")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    train_json_path = os.path.join(args.output_path, "train_samples.json")
    val_json_path = os.path.join(args.output_path, "val_samples.json")
    test_json_path = os.path.join(args.output_path, "test_samples.json")
    get_samples(train_json_path=train_json_path, val_json_path=val_json_path, test_json_path=test_json_path, dataset_path=args.dataset_path, seed=args.seed)
    print("Done!")
