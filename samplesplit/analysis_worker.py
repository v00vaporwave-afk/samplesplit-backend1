from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np


INSTRUMENTS = [
    ("Vocals", "🎤", "Vocals", "a clear vocal singing performance"),
    ("Drum Kit", "🥁", "Drums", "an acoustic or electronic drum kit"),
    ("Bass", "🎸", "Bass", "a bass guitar or synthesized bass line"),
    ("Piano", "🎹", "Instruments", "a piano or keyboard performance"),
    ("Guitar", "🎸", "Instruments", "an acoustic or electric guitar performance"),
    ("Synth", "🎛️", "Instruments", "a synthesizer lead or electronic instrument"),
    ("Strings", "🎻", "Instruments", "orchestral string instruments"),
    ("Brass", "🎺", "Instruments", "brass instruments such as trumpet or trombone"),
    ("Woodwinds", "🪈", "Instruments", "woodwind instruments such as flute or saxophone"),
    ("Pads", "🌌", "Instruments", "a sustained atmospheric synthesizer pad"),
    ("FX", "✨", "Other", "sound effects and production effects"),
    ("Ambient", "🎵", "Other", "ambient texture or atmospheric background audio"),
    ("Percussion", "🪘", "Drums", "hand percussion or auxiliary percussion"),
    ("Other", "◇", "Other", "other musical audio"),
]
MODEL_ID = "laion/clap-htsat-unfused"


def strongest_excerpt(path: Path, sample_rate: int = 48_000, seconds: int = 10) -> tuple[np.ndarray, float]:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if audio.size == 0:
        return np.zeros(sample_rate, dtype=np.float32), 0.0
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    window = sample_rate * seconds
    if audio.size <= window:
        return audio.astype(np.float32), rms
    hop = sample_rate
    energies = [np.mean(np.square(audio[start:start + window]), dtype=np.float64) for start in range(0, audio.size - window + 1, hop)]
    start = int(np.argmax(energies)) * hop
    return audio[start:start + window].astype(np.float32), rms


def classify(paths: list[Path]) -> dict:
    import torch
    from transformers import ClapModel, ClapProcessor

    clips, levels = zip(*(strongest_excerpt(path) for path in paths))
    force_download_check = os.getenv("SAMPLESPLIT_ALLOW_MODEL_DOWNLOAD") == "1"
    try:
        processor = ClapProcessor.from_pretrained(MODEL_ID, local_files_only=not force_download_check)
        model = ClapModel.from_pretrained(MODEL_ID, local_files_only=not force_download_check).eval()
    except OSError:
        processor = ClapProcessor.from_pretrained(MODEL_ID)
        model = ClapModel.from_pretrained(MODEL_ID).eval()
    prompts = [prompt for _, _, _, prompt in INSTRUMENTS]
    inputs = processor(text=prompts, audios=list(clips), sampling_rate=48_000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        probabilities = model(**inputs).logits_per_audio.softmax(dim=-1).cpu().numpy()
    results = {}
    for path, level, scores in zip(paths, levels, probabilities):
        index = int(np.argmax(scores)) if level >= 1e-5 else len(INSTRUMENTS) - 1
        name, icon, group, _ = INSTRUMENTS[index]
        results[path.stem] = {
            "name": name,
            "icon": icon,
            "group": group,
            "confidence": round(float(scores[index]), 4) if level >= 1e-5 else 1.0,
        }
    return {"instruments": results, "model": MODEL_ID}


def music_analysis(path: Path) -> dict:
    audio, sample_rate = librosa.load(path, sr=22_050, mono=True)
    if audio.size == 0:
        return {"bpm": None, "key": None}
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
    bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    profile = np.mean(chroma, axis=1)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    scores = []
    for tonic in range(12):
        scores.append((float(np.corrcoef(profile, np.roll(major, tonic))[0, 1]), tonic, "major"))
        scores.append((float(np.corrcoef(profile, np.roll(minor, tonic))[0, 1]), tonic, "minor"))
    score, tonic, mode = max(scores)
    notes = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]
    return {
        "bpm": round(bpm, 1) if bpm > 0 else None,
        "key": f"{notes[tonic]} {mode}" if np.isfinite(score) and score > 0.15 else None,
        "key_confidence": round(max(0.0, min(1.0, (score + 1) / 2)), 3) if np.isfinite(score) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("classify", "music"))
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    paths = [Path(value) for value in args.paths]
    result = classify(paths) if args.mode == "classify" else music_analysis(paths[0])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
