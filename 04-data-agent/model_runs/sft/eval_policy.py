"""Select completed epochs without evaluating intermediate recovery checkpoints."""
import math


def completed_epoch(epoch):
    if epoch is None or epoch < 1:
        return None
    nearest = round(epoch)
    return nearest if math.isclose(epoch, nearest, rel_tol=0, abs_tol=1e-8) else None


def should_evaluate(step, epoch, interval, final):
    if interval is not None:
        return step % interval == 0 or final
    return completed_epoch(epoch) is not None


def environment_capacity(concurrency):
    return max(128, concurrency + 8)
