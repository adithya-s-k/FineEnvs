# Three standalone environments

[Read the article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) · [Collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212) · [Tutorial](../README.md)

Each directory is a complete Docker Space. Its local files are the files deployed to the Hub. There is no shared environment implementation or generated application bundle.

| Environment | Local source | Space |
|---|---|---|
| SETA whitebox | [whitebox/](whitebox/) | [Open UI](https://fineenvs-smoldataenv-multi-harness-whitebox.hf.space/web/) |
| Native OpenCode | [opencode/](opencode/) | [Open UI](https://fineenvs-smoldataenv-multi-harness-opencode.hf.space/web/) |
| Harbor multi-harness | [harbor/](harbor/) | [Open UI](https://fineenvs-smoldataenv-multi-harness-harbor.hf.space/web/) |

The three Spaces are listed in the [SmolDataEnvs collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-6ab4f2f6e09b7cb872ebc867) and the [SmolDataEnvs Multi-harness RL collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212).

Each contains its own `Dockerfile`, `requirements.txt`, `install.sh`, `start.sh`, `prepare.py`, task manifests, server, and environment package. To run one, enter its directory and follow its README. To deploy it, run its `deploy.py`; it uploads that directory directly. The same Docker build works locally and on a Space.

The Harbor app uses OpenEnv main's rollout and live-trace UI. Opening the Space redirects to `/web/`; `/docs`, the Task API, and capture endpoints remain available. Its Space card enables Hugging Face sign-in and provider selection.

`jobs/run.py` starts these same installed packages inside a GPU allocation. The training scripts import their corresponding environment package. A Space serves the environment; the model can run elsewhere and the task sandboxes run in Daytona.

Native OpenCode's upstream SDK is deprecated. It remains a separate environment here because the comparison explicitly includes the native interface.

[VALIDATION.md](../VALIDATION.md) records tested deployments and GPU checks. Earlier deployment files are preserved in ignored archives and Hub history.
