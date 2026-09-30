"""Check epoch saves with actual Trainer accumulation and a partial last batch."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback

from eval_policy import completed_epoch, should_evaluate, environment_capacity


def test_epoch_policy():
    assert not should_evaluate(50, 0.4, None, False)
    assert not should_evaluate(100, 0.9, None, False)
    assert not should_evaluate(120, 1.2, None, True)
    assert should_evaluate(114, 1.0, None, False)
    assert should_evaluate(228, 2.0, None, True)
    assert should_evaluate(2, 0.25, 2, False)
    assert environment_capacity(100) == 128

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, input_ids, labels=None):
            loss = ((input_ids.float() * self.weight - labels.float()) ** 2).mean()
            return {'loss': loss, 'logits': input_ids.float() * self.weight}

    saved, requested = [], []
    class Policy(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if completed_epoch(state.epoch) is not None:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            saved.append(state.global_step)
            if should_evaluate(state.global_step, state.epoch, None, control.should_training_stop):
                requested.append((state.global_step, completed_epoch(state.epoch)))

    with TemporaryDirectory() as output:
        trainer = Trainer(model=TinyModel(),
            train_dataset=Dataset.from_dict({'input_ids': [[1.0]] * 7, 'labels': [[2.0]] * 7}),
            args=TrainingArguments(output_dir=output, use_cpu=True, num_train_epochs=2,
                per_device_train_batch_size=1, gradient_accumulation_steps=3,
                save_strategy='steps', save_steps=2, report_to='none', disable_tqdm=True),
            callbacks=[Policy()])
        trainer.train()
    assert saved == [2, 3, 4, 6], saved
    assert requested == [(3, 1), (6, 2)], requested
    proof = {'passed': True, 'native_trainer': True, 'examples_per_epoch': 7,
             'gradient_accumulation': 3, 'saved_steps': saved, 'eval_requests': requested,
             'partial_epoch_stop_is_not_evaluated': True, 'production_concurrency': 100,
             'environment_capacity': 128, 'load_test_at_100': False}
    return proof


if __name__ == '__main__':
    import sys
    proof = test_epoch_policy()
    Path(sys.argv[1]).write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))
