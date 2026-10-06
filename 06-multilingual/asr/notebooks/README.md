# Notebook

[`07_multilingual_asr.ipynb`](./07_multilingual_asr.ipynb) runs on a CPU against the hosted Space. It
plays a held-out Kannada clip, scores an answer, and shows what Gemma 4 wrote for that clip before and
after training, scored live. It then plots the whole run from the published results. Two optional
cells run the trained adapter on a GPU and start a short training run.

```bash
cd 06-multilingual/asr
uv run --frozen --project envs/multilingual_asr --with jupyterlab --with matplotlib jupyter lab
```

The saved outputs come from a real run against the Space.
