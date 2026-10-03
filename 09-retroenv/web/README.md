# RetroEval web app

A static, dependency-free explanation and trajectory explorer for the frozen
RetroEval v1 benchmark.

Regenerate the public-safe data bundle after benchmark results change:

```bash
uv run python web/build_data.py
```

Open `web/index.html` directly, or serve the project locally:

```bash
uv run python -m http.server 8080 --directory web
```

Then visit <http://localhost:8080>.

The generated `data.js` contains public tasks, model outputs, compact tool
trajectories, and verifier scores. It deliberately excludes private reference
routes and stock contents.
