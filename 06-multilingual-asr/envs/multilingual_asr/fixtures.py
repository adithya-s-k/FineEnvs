"""Synthetic speech-shaped fixtures. Transport and scoring only; never an ASR benchmark."""

import io

from .data.schema import SPLITS

SAMPLING_RATE = 16_000

# Two space-delimited languages and one character-scored one, so a fixture run exercises
# both error units rather than only the word path.
FIXTURE_LANGUAGES = {
    "en_us": ("English", ["the quick brown fox", "a second short utterance"]),
    "hi_in": ("Hindi", ["यह एक वाक्य है", "दूसरा छोटा वाक्य"]),
    "cmn_hans_cn": ("Mandarin", ["这是一个句子", "第二个短句"]),
}


def tone(seconds, frequency, subtype="FLOAT"):
    """Write a WAV in the same encoding FLEURS publishes (16 kHz mono IEEE float)."""
    import numpy
    import soundfile

    samples = numpy.arange(int(seconds * SAMPLING_RATE), dtype="float32")
    wave_data = 0.35 * numpy.sin(2 * numpy.pi * frequency * samples / SAMPLING_RATE)
    buffer = io.BytesIO()
    soundfile.write(buffer, wave_data, SAMPLING_RATE, format="WAV", subtype=subtype)
    return buffer.getvalue()


def fixture_rows(language):
    name, sentences = FIXTURE_LANGUAGES[language]
    for index, split in enumerate(SPLITS):
        for offset, sentence in enumerate(sentences):
            seconds = 1.0 + 0.25 * offset
            yield {
                "id": index * 100 + offset,
                "split": split,
                "num_samples": int(seconds * SAMPLING_RATE),
                "audio": {"bytes": tone(seconds, 220 + 40 * offset), "path": "a.wav"},
                "transcription": sentence,
                "raw_transcription": sentence[0].upper() + sentence[1:] + ".",
                "language": name,
                "gender": offset % 2,
            }


def make_fixture(directory, languages=tuple(FIXTURE_LANGUAGES)):
    from .data.prepare import write_snapshot

    return write_snapshot(
        directory,
        ((language, fixture_rows(language)) for language in languages),
        source="synthetic-fixture",
        revision="synthetic-fixture",
        languages=list(languages),
    )
