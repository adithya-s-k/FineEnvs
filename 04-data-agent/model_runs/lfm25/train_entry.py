"""Reuse the qualified finite task scheduler with model-specific checkpoint metadata."""
import sys
from pathlib import Path
import atomic_rollouts as atomic
from hard_curriculum_train import FiniteWorker, FiniteTrainer
from checkpoint_artifacts import mark_saved
import train_harbor_multi


class LFMTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        from transformers import AutoTokenizer
        from trl.chat_template_utils import lfm2_2_5_template
        from copy import deepcopy
        tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
        tokenizer.response_template = deepcopy(lfm2_2_5_template)
        return tokenizer

if __name__ == '__main__':
    from packing import install
    install()
    train_harbor_multi.AutoTokenizer = LFMTokenizer
    atomic.AtomicHarnessWorker = FiniteWorker
    atomic.AtomicRolloutTrainer = FiniteTrainer
    original = FiniteTrainer.train
    def train(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self._save_checkpoint(self.model, None)
        model = sys.argv[sys.argv.index('--model') + 1]
        revision = sys.argv[sys.argv.index('--model-revision') + 1]
        mark_saved(Path(self.args.output_dir) / f'checkpoint-{self.state.global_step}',
                   self.state.global_step, model, revision, final=True)
        return result
    FiniteTrainer.train = train
    train_harbor_multi.main()
