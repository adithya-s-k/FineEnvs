# Three standalone environments

Each directory is a complete Docker Space. Its local files are the files deployed to the Hub. There is no shared environment implementation or generated application bundle.

| Environment | Local source | Space |
|---|---|---|
| SETA whitebox | [whitebox/](whitebox/) | [Open UI](https://fineenvs-data-agent-seta-whitebox-env.hf.space/web/) |
| Native OpenCode | [opencode/](opencode/) | [Open UI](https://fineenvs-data-agent-blackbox-opencode-env.hf.space/web/) |
| Harbor multi-harness | [harbor/](harbor/) | [Open UI](https://fineenvs-data-agent-blackbox-harbor-env.hf.space/web/) |

Each contains its own `Dockerfile`, `requirements.txt`, `install.sh`, `start.sh`, `prepare.py`, task manifests, server, and environment package. To run one, enter its directory and follow its README. To deploy it, run its `deploy.py`; it uploads that directory directly. The same Docker build works locally and on a Space.

The Harbor app uses OpenEnv main's rollout and live-trace UI. Opening the Space redirects to `/web/`; `/docs`, the Task API, and capture endpoints remain available. Its Space card enables Hugging Face sign-in and provider selection.

`jobs/run.py` starts these same installed packages inside a GPU allocation. The training scripts import their corresponding environment package. A Space serves the environment; the model can run elsewhere and the task sandboxes run in Daytona.

Native OpenCode's upstream SDK is deprecated. It remains a separate environment here because the comparison explicitly includes the native interface.

[VALIDATION.md](../VALIDATION.md) records tested deployments and GPU checks. Earlier deployment files are preserved in ignored archives and Hub history.
