import io

import numpy
import pytest
import soundfile
from multilingual_asr.data.tasks import SAMPLING_RATE
from multilingual_asr.fixtures import tone
from multilingual_asr.training import decode_audio


def wav(seconds=0.25, rate=SAMPLING_RATE, channels=1, subtype="FLOAT"):
    samples = numpy.linspace(-0.5, 0.5, int(seconds * rate), dtype="float32")
    data = samples if channels == 1 else numpy.stack([samples] * channels, axis=1)
    buffer = io.BytesIO()
    soundfile.write(buffer, data, rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


def test_decodes_the_float_wav_fleurs_actually_publishes():
    """FLEURS is IEEE float, not PCM; the standard library rejects that subtype."""
    audio = decode_audio(wav(subtype="FLOAT"))
    assert audio.dtype == numpy.float32
    assert audio.ndim == 1 and len(audio) == int(0.25 * SAMPLING_RATE)
    assert numpy.abs(audio).max() <= 1.0


def test_decodes_pcm_too_so_a_mixed_source_does_not_break_a_run():
    audio = decode_audio(wav(subtype="PCM_16"))
    assert audio.dtype == numpy.float32 and len(audio) == int(0.25 * SAMPLING_RATE)


def test_fixtures_use_the_same_encoding_as_the_corpus():
    """Fixtures that drift from the corpus hide real decoding failures until a GPU run."""
    info = soundfile.info(io.BytesIO(tone(0.1, 220)))
    assert info.samplerate == SAMPLING_RATE
    assert info.channels == 1
    assert info.subtype == "FLOAT"


def test_a_wrong_sample_rate_or_channel_count_is_rejected_not_resampled():
    with pytest.raises(ValueError, match="Expected 16000 Hz"):
        decode_audio(wav(rate=8000))
    with pytest.raises(ValueError, match="Expected mono"):
        decode_audio(wav(channels=2))
