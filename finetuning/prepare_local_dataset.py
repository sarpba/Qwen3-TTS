# coding=utf-8
import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from finetuning.language_utils import canonicalize_language_name
except ImportError:
    from language_utils import canonicalize_language_name


SUPPORTED_LANGUAGES = {
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
    "Hungarian",
}

FILENAME_RE = re.compile(
    r"^(?P<utt>\d+?)_(?P<speaker>SPEAKER_\d+?)_(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)$"
)


@dataclass
class Sample:
    leaf_dir: str
    stem: str
    speaker: str
    audio_path: str
    transcript: str
    language: str
    duration: float


def format_seconds(total_seconds: float) -> str:
    total_seconds = float(total_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build finetuning JSONL from pyannote-style leaf directories."
    )
    parser.add_argument("-i", "--input", required=True, help="Root directory to scan.")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "-t",
        "--transcript-suffix",
        default=".json",
        help=(
            "Suffix appended to the audio basename when locating transcript JSON files. "
            "Default: .json"
        ),
    )
    parser.add_argument(
        "--min-samples-per-reference",
        type=int,
        default=5,
        help="Keep only speaker/reference groups with at least this many audio files (default: 5).",
    )
    parser.add_argument(
        "--max-samples-per-reference",
        type=int,
        default=0,
        help="Keep at most this many audio files per reference group; 0 disables the cap (default: 0).",
    )
    return parser.parse_args()


def iter_leaf_dirs(root_dir: str) -> Iterable[str]:
    for current_root, dirnames, filenames in os.walk(root_dir):
        if dirnames:
            continue
        if any(name.endswith(".flac") for name in filenames):
            yield current_root


def parse_filename(stem: str) -> Optional[Tuple[str, float]]:
    match = FILENAME_RE.match(stem)
    if not match:
        return None
    speaker = match.group("speaker")
    start = float(match.group("start"))
    end = float(match.group("end"))
    return speaker, max(0.0, end - start)


def load_transcript_and_language(json_path: str) -> Tuple[str, Optional[str]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    segments = payload.get("segments", [])
    texts = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if text:
            texts.append(text)
    transcript = " ".join(texts).strip()
    raw_language = payload.get("language")
    return transcript, raw_language


def normalize_language(raw_language: Optional[str]) -> Optional[str]:
    if raw_language is None:
        return None
    language = canonicalize_language_name(raw_language)
    if language not in SUPPORTED_LANGUAGES:
        return None
    return language


def resolve_output_path(output_arg: str) -> str:
    normalized = os.path.abspath(output_arg)
    if output_arg.endswith(os.sep) or os.path.isdir(normalized):
        return os.path.join(normalized, "dataset.jsonl")
    return normalized


def build_leaf_samples(leaf_dir: str, stats: Counter, transcript_suffix: str) -> List[Sample]:
    samples: List[Sample] = []

    for name in sorted(os.listdir(leaf_dir)):
        if not name.endswith(".flac"):
            continue

        stem = os.path.splitext(name)[0]
        parsed = parse_filename(stem)
        if parsed is None:
            stats["skipped_bad_filename"] += 1
            continue

        json_path = os.path.join(leaf_dir, f"{stem}{transcript_suffix}")
        if not os.path.exists(json_path):
            stats["skipped_missing_json"] += 1
            continue

        transcript, raw_language = load_transcript_and_language(json_path)
        if not transcript:
            stats["skipped_empty_transcript"] += 1
            continue

        language = normalize_language(raw_language)
        if language is None:
            stats["skipped_unsupported_language"] += 1
            continue

        speaker, duration = parsed
        samples.append(
            Sample(
                leaf_dir=leaf_dir,
                stem=stem,
                speaker=speaker,
                audio_path=os.path.join(leaf_dir, name),
                transcript=transcript,
                language=language,
                duration=duration,
            )
        )

    return samples


def deterministic_choice(leaf_dir: str, speaker: str, candidates: List[Sample]) -> Sample:
    preferred = [sample for sample in candidates if 1.0 <= sample.duration <= 5.0]
    if preferred:
        pool = preferred
    else:
        ordered = sorted(candidates, key=lambda sample: (sample.duration, sample.stem))
        cutoff = max(1, (len(ordered) + 3) // 4)
        pool = ordered[:cutoff]

    pool = sorted(pool, key=lambda sample: (sample.duration, sample.stem))
    digest = hashlib.sha256(f"{leaf_dir}::{speaker}".encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(pool)
    return pool[index]


def deterministic_sample_subset(
    leaf_dir: str,
    speaker: str,
    candidates: List[Sample],
    max_samples: int,
) -> List[Sample]:
    if max_samples <= 0 or len(candidates) <= max_samples:
        return sorted(candidates, key=lambda sample: sample.stem)

    ranked = []
    for sample in candidates:
        digest = hashlib.sha256(
            f"{leaf_dir}::{speaker}::{sample.stem}".encode("utf-8")
        ).hexdigest()
        ranked.append((digest, sample.stem, sample))

    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = [item[2] for item in ranked[:max_samples]]
    return sorted(selected, key=lambda sample: sample.stem)


def build_records(
    root_dir: str,
    transcript_suffix: str,
    min_samples_per_reference: int,
    max_samples_per_reference: int,
) -> Tuple[List[Dict], Counter, Dict[str, List[Sample]]]:
    stats = Counter()
    records: List[Dict] = []
    ref_groups: Dict[str, List[Sample]] = defaultdict(list)

    for leaf_dir in sorted(iter_leaf_dirs(root_dir)):
        stats["leaf_dirs_seen"] += 1
        samples = build_leaf_samples(leaf_dir, stats, transcript_suffix)
        if not samples:
            stats["leaf_dirs_without_valid_samples"] += 1
            continue

        grouped: Dict[str, List[Sample]] = defaultdict(list)
        for sample in samples:
            grouped[sample.speaker].append(sample)

        ref_by_speaker: Dict[str, Sample] = {}
        kept_samples_by_speaker: Dict[str, List[Sample]] = {}
        for speaker, speaker_samples in grouped.items():
            if len(speaker_samples) < min_samples_per_reference:
                stats["skipped_small_reference_groups"] += 1
                stats["skipped_small_reference_group_samples"] += len(speaker_samples)
                continue
            kept_samples = deterministic_sample_subset(
                leaf_dir,
                speaker,
                speaker_samples,
                max_samples_per_reference,
            )
            if len(kept_samples) < len(speaker_samples):
                stats["capped_reference_groups"] += 1
                stats["capped_reference_group_samples"] += len(speaker_samples) - len(kept_samples)
            kept_samples_by_speaker[speaker] = kept_samples
            ref_by_speaker[speaker] = deterministic_choice(leaf_dir, speaker, kept_samples)
            stats["speakers_seen"] += 1

        for speaker, kept_samples in kept_samples_by_speaker.items():
            ref_sample = ref_by_speaker[speaker]
            for sample in kept_samples:
                ref_groups[ref_sample.audio_path].append(sample)
                records.append(
                    {
                        "audio": sample.audio_path,
                        "text": sample.transcript,
                        "ref_audio": ref_sample.audio_path,
                        "ref_text": ref_sample.transcript,
                        "language": sample.language,
                    }
                )
                stats["records_written"] += 1
                if sample.audio_path == ref_sample.audio_path:
                    stats["records_using_self_reference"] += 1

    return records, stats, ref_groups


def print_summary(records: List[Dict], stats: Counter, ref_groups: Dict[str, List[Sample]], output_path: str) -> None:
    print(f"Wrote {len(records)} records to {output_path}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")

    if not records:
        return

    group_sizes = [len(samples) for samples in ref_groups.values()]
    group_durations = [sum(sample.duration for sample in samples) for samples in ref_groups.values()]
    total_duration = sum(group_durations)
    avg_audio_duration = total_duration / len(records)
    avg_group_size = sum(group_sizes) / len(group_sizes)
    avg_group_duration = total_duration / len(group_durations)
    language_counts = Counter(record["language"] for record in records)

    print("summary.unique_reference_groups:", len(ref_groups))
    print(
        "summary.audios_per_reference:"
        f" min={min(group_sizes)} max={max(group_sizes)} avg={avg_group_size:.2f}"
    )
    print(
        "summary.total_audio_duration_per_reference:"
        f" min={format_seconds(min(group_durations))}"
        f" max={format_seconds(max(group_durations))}"
        f" avg={format_seconds(avg_group_duration)}"
    )
    print(
        "summary.total_useful_audio:"
        f" rows={len(records)} duration={format_seconds(total_duration)}"
    )
    print(f"summary.average_audio_length: {avg_audio_duration:.3f}s")
    print("summary.language_distribution:")
    for language, count in sorted(language_counts.items()):
        print(f"  {language}: {count}")


def main():
    args = parse_args()
    records, stats, ref_groups = build_records(
        args.input,
        args.transcript_suffix,
        args.min_samples_per_reference,
        args.max_samples_per_reference,
    )
    output_path = resolve_output_path(args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print_summary(records, stats, ref_groups, output_path)


if __name__ == "__main__":
    main()
