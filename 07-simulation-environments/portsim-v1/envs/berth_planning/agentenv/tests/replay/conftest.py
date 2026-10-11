import pyarrow.parquet as pq
import pytest
from huggingface_hub import hf_hub_download

DATASET, REVISION = "FineEnvs/PortSimEnv", "5304899c94f5fbe16b7c6b7fbce98fd90dc9899b"


@pytest.fixture(scope="session")
def episodes() -> list[dict]:
    """The 300 published dock-eval50 episodes (6 models x 50 tasks), fetched once into the Hugging Face cache."""
    path = hf_hub_download(DATASET, "rollouts/eval.parquet", repo_type="dataset", revision=REVISION)
    return pq.read_table(path).to_pylist()
