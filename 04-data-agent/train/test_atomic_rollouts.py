import asyncio
import pickle
import queue
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from atomic_rollouts import (
    AtomicHarnessLoop,
    AtomicRolloutDataset,
    AtomicRolloutTrainer,
    RolloutBundle,
    pack_rows,
)
from transformers import Trainer, TrainingArguments
from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample
from trl.experimental.async_grpo.openenv_harness import _HarnessRolloutLoop


def row(ids, mask=None, group=0, version=1, advantage=0.7):
    return RolloutSample(
        [],
        [],
        ids,
        mask or [0] + [1] * (len(ids) - 1),
        [-0.8] * len(ids),
        advantage,
        version,
        group,
        {"reward": 1.0},
    )


def worker(bundles, version=1):
    q = queue.Queue()
    for b in bundles:
        q.put(b)
    return SimpleNamespace(
        rollout_buffer=q,
        model_version=version,
        check_health=lambda timeout: pytest.fail("Unexpected empty queue"),
    )


def test_bundle_survives_worker_process_pickle():
    original = RolloutBundle([row([1, 2, 3]), row([1, 4])], "episode")
    copy = pickle.loads(pickle.dumps(original))
    assert copy.rollout_id == "episode" and len(copy.rows) == 2
    assert copy.model_version == 1 and copy.group_id == 0


def test_native_scored_rows_keep_their_rollout_and_advantage(monkeypatch):
    original = [row([1, 2]), row([1, 3]), row([1, 4], advantage=-0.2)]

    async def score(self, group):
        return original

    monkeypatch.setattr(_HarnessRolloutLoop, "_score_group", score)
    group = SimpleNamespace(
        group_id=0,
        completions_sequences=[
            [SimpleNamespace(rollout_id="a"), SimpleNamespace(rollout_id="a")],
            [],
            [SimpleNamespace(rollout_id="c")],
        ],
    )
    bundles = asyncio.run(
        AtomicHarnessLoop._score_group(object.__new__(AtomicHarnessLoop), group)
    )
    assert [b.rollout_id for b in bundles] == ["a", "c"]
    assert bundles[0].rows == original[:2] and bundles[1].rows == original[2:]
    assert bundles[1].advantage == -0.2


def test_forked_rollout_is_admitted_whole_and_forwarded_without_token_changes():
    bundle = RolloutBundle(
        [row([1, 2, 3]), row([4, 5, 6, 7, 8, 9]), row([1, 4])], "fork"
    )
    data = AtomicRolloutDataset(worker([bundle]), defaultdict(list), 4, 8, 4, 60)
    unit = next(iter(data))
    assert unit["rollouts"] == [bundle]
    packs = list(pack_rows(unit["rollouts"], 4, 8))
    result = [r for pack in packs for r in pack]
    assert [r["input_ids"] for r in result] == [r.input_ids for r in bundle.rows]
    assert [r["completion_mask"] for r in result] == [
        r.completion_mask for r in bundle.rows
    ]
    assert [r["old_log_probs"] for r in result] == [
        r.old_log_probs for r in bundle.rows
    ]


def test_pending_rollout_is_rechecked_for_staleness_and_never_partially_dropped():
    a = RolloutBundle([row([1, 2, 3])], "a")
    b = RolloutBundle([row([1, 2, 3]), row([1, 4])], "b")
    c = RolloutBundle([row([1, 2, 3, 4], version=6)], "c")
    w = worker([a, b, c])
    metrics = defaultdict(list)
    data = AtomicRolloutDataset(w, metrics, 4, 8, 4, 60)
    it = iter(data)
    assert [x.rollout_id for x in next(it)["rollouts"]] == ["a"]
    w.model_version = 6
    assert [x.rollout_id for x in next(it)["rollouts"]] == ["c"]
    assert metrics["admission/stale_rollouts_dropped_total"] == [1]
    assert metrics["sample/dropped_stale_total"] == [2]


def test_context_overflow_fails_instead_of_silently_losing_rows():
    data = AtomicRolloutDataset(
        worker([RolloutBundle([row(list(range(10)))], "long")]),
        defaultdict(list),
        4,
        8,
        4,
        60,
    )
    with pytest.raises(RuntimeError, match="refusing to discard"):
        next(iter(data))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.02))

    def forward(self, input_ids, **kwargs):
        lp = self.theta - 0.8 + input_ids[:, 1:] * 0.003
        return {"log_probs": lp, "entropy": torch.ones_like(lp)}


def native_loss_trainer():
    trainer = object.__new__(AtomicRolloutTrainer)
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.aux_loss_enabled = False
    trainer.accelerator = SimpleNamespace(
        num_processes=1,
        reduce=lambda value, reduction: value,
        gather=lambda value: value,
    )
    trainer._metrics = {"train": defaultdict(list)}
    trainer.current_gradient_accumulation_steps = 4
    for name in [
        "_step_forward_tokens",
        "_step_trained_tokens",
        "_step_seq_len_weighted",
        "_step_samples",
        "_step_forward_s",
    ]:
        setattr(trainer, name, 0.0)
    return trainer


@pytest.mark.parametrize("target", [4, 8, 100])
def test_actual_trl_loss_gradient_is_invariant_to_fork_packing(target):
    bundles = [
        RolloutBundle(
            [row([1, 2, 3], [0, 0, 1]), row([1, 4, 5, 6], advantage=-0.2)], "a"
        ),
        RolloutBundle([row([1, 7, 8, 9], [0, 0, 1, 1])], "b"),
    ]
    trainer = native_loss_trainer()
    model = ToyModel()
    denominator = sum(sum(r.completion_mask) for b in bundles for r in b.rows)
    trainer._atomic_normalization_tokens = denominator
    collator = DataCollatorForRollout(0)
    total = torch.zeros(())
    for packed in pack_rows(bundles, target, 16):
        loss = trainer.compute_loss(model, collator([[packed]]))
        loss.backward()
        total += loss.detach()
    expected_model = ToyModel()
    expected = torch.zeros(())
    for b in bundles:
        for r in b.rows:
            ids = torch.tensor(r.input_ids[1:])
            mask = torch.tensor(r.completion_mask[1:])
            expected += (
                -torch.exp(expected_model.theta + ids * 0.003) * r.advantage * mask
            ).sum()
    expected /= denominator
    expected.backward()
    torch.testing.assert_close(total, expected.detach())
    torch.testing.assert_close(model.theta.grad, expected_model.theta.grad)
    assert trainer._step_trained_tokens == denominator


@pytest.mark.parametrize("credit_limit", [0, 8])
def test_real_hf_optimizer_loop_consumes_all_forks_before_updating(
    tmp_path, credit_limit
):
    import json

    from atomic_rollouts import AtomicAdmissionCallback

    class CPUTrainer(AtomicRolloutTrainer):
        _inner_training_loop = Trainer._inner_training_loop
        log = Trainer.log

        def __init__(self):
            args = TrainingArguments(
                output_dir=str(tmp_path),
                use_cpu=True,
                max_steps=2,
                gradient_accumulation_steps=4,
                learning_rate=1e-3,
                report_to=[],
                save_strategy="no",
                logging_strategy="no",
                disable_tqdm=True,
            )
            args.token_budget = 4
            args.max_staleness = 4
            args.heartbeat_stale_after_s = 60
            Trainer.__init__(
                self,
                model=ToyModel(),
                args=args,
                compute_loss_func="native AsyncGRPO disables HF loss scaling",
            )
            self.model_accepts_loss_kwargs = False
            self.processing_class = SimpleNamespace(pad_token_id=0)
            self.rollout_worker = worker(
                [
                    RolloutBundle(
                        [row([1, 2, 3], group=i // 4), row([1, 4, 5], group=i // 4)],
                        f"episode-{i}",
                    )
                    for i in range(8)
                ]
            )
            self.released_rollouts = []
            if credit_limit:
                self.rollout_worker.max_outstanding_rollouts = credit_limit
                self.rollout_worker.num_generations = 4
                self.rollout_worker.release_rollouts = self.released_rollouts.append
            self.max_row_tokens = 8
            self._trained_groups = set()
            self._groups_before_resume = 0
            self._metrics = {"train": defaultdict(list)}
            self.epsilon_low = self.epsilon_high = 0.2
            self.aux_loss_enabled = False
            self.admission_dir = tmp_path
            self._atomic_finished = []
            for name in [
                "_step_forward_tokens",
                "_step_trained_tokens",
                "_step_seq_len_weighted",
                "_step_samples",
                "_step_forward_s",
                "_step_microbatches",
                "_current_train_step_time",
            ]:
                setattr(self, name, 0.0)
            self.add_callback(AtomicAdmissionCallback(self))

    trainer = CPUTrainer()
    before = trainer.model.theta.detach().clone()
    trainer.train()
    assert trainer.state.global_step == 2
    assert trainer._step_microbatches == 16
    assert not torch.equal(before, trainer.model.theta.detach())
    receipts = [
        json.loads(line)
        for line in (tmp_path / "optimizer_rollouts.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in receipts] == [1, 2]
    assert all(
        len(r["rollouts"]) == 4 and r["all_admitted_rows_consumed"] for r in receipts
    )
    assert sum(x["rows"] for r in receipts for x in r["rollouts"]) == 16
    assert len({x["rollout_id"] for r in receipts for x in r["rollouts"]}) == 8
    assert trainer.released_rollouts == ([4, 4] if credit_limit else [])


def _use_spawned_credits(channel, result):
    first = channel.reserve_group(4)
    second = channel.reserve_group(4)
    channel.release(2)
    result.put((first, second, channel.credits.value))


def test_credit_queue_uses_native_spawn_ipc():
    import multiprocessing

    from atomic_rollouts import CreditQueue

    ctx = multiprocessing.get_context("spawn")
    channel = CreditQueue(ctx.Queue(), ctx.Value("i", 6), 6)
    result = ctx.Queue()
    child = ctx.Process(target=_use_spawned_credits, args=(channel, result))
    child.start()
    try:
        assert result.get(timeout=150) == (True, False, 4)
        child.join(timeout=15)
        assert child.exitcode == 0 and channel.credits.value == 4
        with pytest.raises(RuntimeError, match="more than once"):
            channel.release(3)
    finally:
        if child.is_alive():
            child.terminate()
            child.join()
        channel.close()
        result.close()


def test_generation_reserves_complete_groups_and_tags_actual_dispatch(monkeypatch):
    import multiprocessing

    from atomic_rollouts import CreditQueue
    from trl.experimental.async_grpo.async_rollout_worker import RolloutGroup

    ctx = multiprocessing.get_context("spawn")
    channel = CreditQueue(queue.Queue(), ctx.Value("i", 4), 4)
    loop = object.__new__(AtomicHarnessLoop)
    loop.rollout_buffer = channel
    loop.num_generations = 2
    loop._model_version_value = SimpleNamespace(value=1)
    loop._stop_event = asyncio.Event()
    called = []

    async def generate(self, prompt, tool_dict, tools, group_id=0):
        called.append((group_id, self.model_version))
        return ([], [], [SimpleNamespace(rollout_id=f"{group_id}-{prompt}")], 0, 0, 1.0)

    async def score(self, group):
        assert group.model_version == 5
        return [row([1, 2], version=5) for _ in group.completions_sequences]

    monkeypatch.setattr(_HarnessRolloutLoop, "_generate_one", generate)
    monkeypatch.setattr(_HarnessRolloutLoop, "_score_group", score)

    async def exercise():
        tasks = [
            asyncio.create_task(loop._generate_one(i, {}, [], i // 2)) for i in range(6)
        ]
        await asyncio.sleep(0.15)
        assert len(called) == 4 and channel.credits.value == 0
        assert sum(task.done() for task in tasks) == 4
        loop._model_version_value.value = 5
        channel.release(2)  # two whole rollouts consumed by the optimizer
        results = await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert called[-2:] == [(2, 5), (2, 5)]
        group = RolloutGroup(
            [], {}, [], [], [r[2] for r in results[-2:]], [], [], 1, 2, [], []
        )
        bundles = await loop._score_group(group)
        assert [b.model_version for b in bundles] == [5, 5]
        assert channel.credits.value == 0  # scoring alone does not release credits

    asyncio.run(exercise())


def test_empty_generation_returns_its_reserved_credit(monkeypatch):
    import multiprocessing

    from atomic_rollouts import CreditQueue

    ctx = multiprocessing.get_context("spawn")
    loop = object.__new__(AtomicHarnessLoop)
    loop.rollout_buffer = CreditQueue(queue.Queue(), ctx.Value("i", 4), 4)
    loop.num_generations = 2
    loop._model_version_value = SimpleNamespace(value=1)
    loop._stop_event = asyncio.Event()

    async def empty(*args, **kwargs):
        return ([], [], [], 0, 0, None)

    monkeypatch.setattr(_HarnessRolloutLoop, "_generate_one", empty)

    async def exercise():
        await asyncio.gather(
            loop._generate_one(0, {}, [], 0), loop._generate_one(1, {}, [], 0)
        )
        assert loop.rollout_buffer.credits.value == 4

    asyncio.run(exercise())


def test_small_rows_cannot_hold_all_credits_waiting_for_token_target():
    # Sixteen outstanding generations, G8 and GAS4: each unit may consume at
    # most two rollouts, reserving room for a complete new group until update.
    bundles = [RolloutBundle([row([1, 2])], f"episode-{i}") for i in range(8)]
    data = AtomicRolloutDataset(
        worker(bundles),
        defaultdict(list),
        40960,
        131072,
        4,
        60,
        max_rollouts_per_unit=2,
    )
    it = iter(data)
    batches = [next(it) for _ in range(4)]
    assert [len(b["rollouts"]) for b in batches] == [2, 2, 2, 2]


def test_multiple_stale_rollouts_release_credits_and_log_exact_totals(tmp_path):
    import json

    from trl.experimental.async_grpo.async_grpo_trainer import _reduce_metric

    old = [RolloutBundle([row([1, 2])] * n, f"old-{n}") for n in (2, 3)]
    new = RolloutBundle([row([1, 2, 3, 4], version=6)], "new")
    w = worker(old + [new], version=6)
    released = []
    w.release_rollouts = released.append
    metrics = defaultdict(list)
    path = tmp_path / "rejections.jsonl"
    data = AtomicRolloutDataset(
        w, metrics, 4, 8, 4, 60, rejection_path=path, group_offset=81
    )
    assert next(iter(data))["rollouts"] == [new]
    assert released == [1, 1]
    assert (
        _reduce_metric(
            "admission/stale_rollouts_dropped_total",
            metrics["admission/stale_rollouts_dropped_total"],
        )
        == 2
    )
    assert (
        _reduce_metric(
            "sample/dropped_stale_total", metrics["sample/dropped_stale_total"]
        )
        == 5
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["rows"] for r in records] == [2, 3]
    assert all(r["group_id"] == 81 for r in records)
