"""Derive ASR tasks from one FLEURS row. References never enter a public payload."""

from .schema import FAMILIES, character_scored, normalize_for_scoring, task_id

SAMPLING_RATE = 16_000
# FLEURS utterances are read sentences; anything far outside this is a source defect
# rather than a hard example, and is excluded rather than quietly scored.
MIN_SECONDS = 0.5
MAX_SECONDS = 60.0

PROMPTS = {
    "transcription": (
        "Transcribe this {name} speech. Return only the transcription, in {name}, "
        "lowercased and without punctuation."
    ),
    "verbatim_transcription": (
        "Transcribe this {name} speech exactly as spoken, in {name}, preserving "
        "capitalization, punctuation, and numerals. Return only the transcription."
    ),
    "language_id": (
        "Identify the language of this speech. Return only its FLEURS code, for "
        "example en_us or hi_in."
    ),
}


def derive_tasks(row, language, revision, *, families=FAMILIES, metadata_only=False):
    """Return every eligible task for one utterance, plus why any were skipped."""
    skipped = {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1
        return [], skipped

    sample_id = row.get("id")
    if sample_id is None:
        return skip("missing_id")
    num_samples = row.get("num_samples") or 0
    duration = num_samples / SAMPLING_RATE
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        return skip("duration_outside_policy")
    audio = row.get("audio") or {}
    media = audio.get("bytes") if not metadata_only else None
    if not metadata_only and not isinstance(media, bytes):
        return skip("missing_audio_bytes")

    references = {
        "transcription": row.get("transcription"),
        "verbatim_transcription": row.get("raw_transcription"),
        "language_id": language,
    }
    tasks = []
    for family in families:
        reference = references.get(family)
        if not isinstance(reference, str) or not reference.strip():
            skipped[f"empty_{family}_reference"] = (
                skipped.get(f"empty_{family}_reference", 0) + 1
            )
            continue
        if family != "language_id" and not normalize_for_scoring(reference, family):
            # Nothing would remain to score against.
            skipped[f"unscorable_{family}_reference"] = (
                skipped.get(f"unscorable_{family}_reference", 0) + 1
            )
            continue
        tasks.append(
            {
                "task_id": task_id(revision, language, row["split"], sample_id, family),
                "language": language,
                "language_name": row.get("language") or language,
                "split": row["split"],
                "family": family,
                "sample_id": int(sample_id),
                "prompt": PROMPTS[family].format(name=row.get("language") or language),
                "reference": reference,
                "media": media,
                "mime": "audio/wav",
                "sampling_rate": SAMPLING_RATE,
                "num_samples": int(num_samples),
                "duration_seconds": round(duration, 3),
                "error_unit": "cer" if character_scored(language) else "wer",
                "gender": row.get("gender"),
            }
        )
    return tasks, skipped
