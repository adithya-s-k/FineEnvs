"""Compare FLA outputs and gradients with the Torch reference on an HF GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def worker():
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    print('GPU', torch.cuda.get_device_name(), 'FLA_TILELANG', os.environ['FLA_TILELANG'], flush=True)
    for length in [63, 256, 2048]:
        torch.manual_seed(17)
        shape = (1, length, 4, 128)
        q = torch.nn.functional.normalize(torch.randn(shape, device='cuda'), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(torch.randn(shape, device='cuda'), dim=-1).bfloat16()
        v = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        g = -torch.rand((1, length, 4), device='cuda')
        beta = torch.rand((1, length, 4), device='cuda', dtype=torch.bfloat16)
        upstream = torch.randn_like(v)
        values = []
        for fn in [torch_chunk_gated_delta_rule, chunk_gated_delta_rule]:
            args = [x.detach().clone().requires_grad_() for x in [q, k, v, g, beta]]
            out, _ = fn(*args)
            (out.float() * upstream).sum().backward()
            torch.cuda.synchronize()
            values.append([out.detach().float()] + [x.grad.detach().float() for x in args])
        ratios = {}
        for name, expected, actual in zip(['output', 'dq', 'dk', 'dv', 'dg', 'dbeta'], *values):
            assert torch.isfinite(actual).all(), name
            ratio = ((actual - expected).square().mean() / expected.square().mean().clamp_min(1e-20)).sqrt().item()
            ratios[name] = ratio
            assert ratio < .05, (length, name, ratio)
        print(json.dumps({'length': length, 'relative_rms_errors': ratios, 'passed': True}), flush=True)


def main():
    if '--worker' in sys.argv:
        worker()
        return
    root = Path('/workspace/kernel-check')
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'requirements.lock'
    override = Path('/bundle/requirements-kernel.lock')
    if override.exists():
        lock.write_bytes(override.read_bytes())
    else:
        with tarfile.open('/bundle/bundle.tar.gz') as archive:
            lock.write_bytes(archive.extractfile('hf/locks/requirements-train.lock').read())
    python = root / '.venv/bin/python'
    subprocess.run(['uv', 'venv', '--python', '3.12', str(python.parent.parent)], check=True)
    subprocess.run(['uv', 'pip', 'sync', '--python', str(python), '--require-hashes', str(lock)], check=True)
    results = []
    for backend in os.environ.get('TEST_BACKENDS', '0,1').split(','):
        env = {**os.environ, 'FLA_TILELANG': backend, 'CUDA_VISIBLE_DEVICES': '0',
               'PYTHONFAULTHANDLER': '1', 'PYTHONUNBUFFERED': '1'}
        result = subprocess.run([str(python), __file__, '--worker'], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        print(result.stdout, flush=True)
        results.append({'FLA_TILELANG': backend, 'returncode': result.returncode})
        print('BACKEND_RESULT', json.dumps(results[-1]), flush=True)
    print('DIAGNOSTIC_RESULTS', json.dumps(results), flush=True)
    if all(x['returncode'] for x in results):
        raise RuntimeError('Neither kernel backend passed')


if __name__ == '__main__':
    main()
