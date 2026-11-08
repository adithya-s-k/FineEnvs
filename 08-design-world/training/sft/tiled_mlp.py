"""Sequence-tiled MLP for long-context training (the TiledMLP idea of Arctic Long Sequence Training).

An MLP (dense or MoE) treats every token on its own, so its forward can run over the sequence in tiles: the forward
keeps only its input, and the backward recomputes and backpropagates one tile at a time. A layer's MLP intermediates
(gate, up, activation, expert buffers: the largest part of a decoder layer's backward peak at long context) then
take one tile's memory instead of the whole sequence's. Gradients for the trainable parameters are returned by the
autograd function itself (each accumulated once per step), so DDP's gradient hooks see the usual single update.
"""

from __future__ import annotations

import re

import torch


class _TiledMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, tile: int, x: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        ctx.fn, ctx.tile, ctx.params = fn, tile, params
        ctx.autocast = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        ctx.save_for_backward(x)
        with torch.no_grad():
            return torch.cat([fn(x[:, i:i + tile].contiguous()) for i in range(0, x.shape[1], tile)], dim=1)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        params = [p for p in ctx.params if p.requires_grad]
        x_grad = torch.empty_like(x) if ctx.needs_input_grad[2] else None
        p_grads = [None] * len(params)
        enabled, dtype = ctx.autocast
        for i in range(0, x.shape[1], ctx.tile):
            xc = x[:, i:i + ctx.tile].contiguous().detach().requires_grad_(x_grad is not None)
            # (identity hooks: the tile's own graph lives only for this loop, so an activation offloader around the
            # training step must not copy it to the CPU and back)
            with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
                with torch.enable_grad(), torch.autocast("cuda", dtype=dtype, enabled=enabled):
                    out = ctx.fn(xc)
                inputs = ([xc] if x_grad is not None else []) + params
                gs = torch.autograd.grad(out, inputs, grad[:, i:i + ctx.tile], allow_unused=True)
            if x_grad is not None:
                x_grad[:, i:i + ctx.tile] = gs[0]
                gs = gs[1:]
            for j, g in enumerate(gs):
                if g is not None:
                    p_grads[j] = g if p_grads[j] is None else p_grads[j] + g
        it = iter(p_grads)
        return (None, None, x_grad, *(next(it) if p.requires_grad else None for p in ctx.params))


class _TiledRows(torch.autograd.Function):
    """`fn(*inputs)` for a row-wise function (each row of the inputs' dim 0 on its own, e.g. a norm), run in tiles of
    rows: the forward keeps only the inputs, the backward recomputes one tile at a time."""

    @staticmethod
    def forward(ctx, fn, tile: int, n: int, *args: torch.Tensor) -> torch.Tensor:
        ctx.fn, ctx.tile, ctx.n, ctx.params = fn, tile, n, args[n:]
        ctx.autocast = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        ctx.save_for_backward(*args[:n])
        with torch.no_grad():
            return torch.cat([fn(*(x[i:i + tile] for x in args[:n])) for i in range(0, args[0].shape[0], tile)])

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        xs = ctx.saved_tensors
        params = [p for p in ctx.params if p.requires_grad]
        need = [ctx.needs_input_grad[3 + k] for k in range(ctx.n)]
        x_grads = [torch.empty_like(x) if nd else None for x, nd in zip(xs, need)]
        p_grads = [None] * len(params)
        enabled, dtype = ctx.autocast
        for i in range(0, xs[0].shape[0], ctx.tile):
            xc = [x[i:i + ctx.tile].detach().requires_grad_(nd) for x, nd in zip(xs, need)]
            with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
                with torch.enable_grad(), torch.autocast("cuda", dtype=dtype, enabled=enabled):
                    out = ctx.fn(*xc)
                inputs = [x for x, nd in zip(xc, need) if nd] + params
                gs = list(torch.autograd.grad(out, inputs, grad[i:i + ctx.tile], allow_unused=True)) if inputs else []
            for k, nd in enumerate(need):
                if nd:
                    x_grads[k][i:i + ctx.tile] = gs.pop(0)
            for j, g in enumerate(gs):
                if g is not None:
                    p_grads[j] = g if p_grads[j] is None else p_grads[j] + g
        it = iter(p_grads)
        return (None, None, None, *x_grads, *(next(it) if p.requires_grad else None for p in ctx.params))


def tile_gated_norms(model: torch.nn.Module, tile: int, pattern: str = r".*linear_attn\.norm") -> int:
    """Run each linear-attention layer's gated RMSNorm (fp32 inside: about seven full-size copies of its input kept
    for the backward) in tiles of `tile` rows while gradients are on, so only its two bf16 inputs are kept."""
    n = 0
    for name, m in model.named_modules():
        if not re.fullmatch(pattern, name):
            continue
        orig = m.forward

        def forward(x, gate, _orig=orig, _m=m):
            if x.shape[0] <= tile or not torch.is_grad_enabled():
                return _orig(x, gate)
            return _TiledRows.apply(_orig, tile, 2, x, gate, *[p for p in _m.parameters() if p.requires_grad])

        m.forward = forward
        n += 1
    return n


def tile_mlps(model: torch.nn.Module, tile: int, pattern: str = r".*language_model\.layers\.\d+\.mlp") -> int:
    """Run every module whose name matches `pattern` (each decoder layer's MLP) in sequence tiles of `tile` tokens
    while gradients are on. Call before or after PEFT: trainable parameters are collected at each call."""
    n = 0
    for name, m in model.named_modules():
        if not re.fullmatch(pattern, name):
            continue
        orig = m.forward

        def forward(x, _orig=orig, _m=m):
            if x.dim() != 3 or x.shape[1] <= tile or not torch.is_grad_enabled():
                return _orig(x)
            return _TiledMLP.apply(_orig, tile, x, *[p for p in _m.parameters() if p.requires_grad])

        m.forward = forward
        n += 1
    return n


def segment_delta_rule(segment: int) -> list[str]:
    """Run the gated delta rule of Qwen3.5-family linear-attention layers over the sequence in segments of `segment`
    tokens while gradients are on, each segment starting from the recurrent state the previous one ended in (exact:
    that state is all the segments share; its gradient flows back through the kernels' final/initial-state grads).
    The kernels' backward then allocates one segment's per-chunk states and gradients instead of the whole sequence's
    (the largest single allocation of the 27B's layer backward at ~245k tokens)."""
    import importlib

    patched = []
    for name in ("qwen3_5", "qwen3_5_moe"):
        try:
            mod = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        except ImportError:
            continue
        orig = mod.torch_chunk_gated_delta_rule

        def seg_rule(q, k, v, *args, _orig=orig, **kw):
            n = q.shape[1]
            if n <= segment or not torch.is_grad_enabled() or kw.get("cu_seqlens") is not None or args:
                return _orig(q, k, v, *args, **kw)
            g, beta, state = kw.pop("g"), kw.pop("beta"), kw.pop("initial_state", None)
            want_final = kw.pop("output_final_state", False)
            outs = []
            for i in range(0, n, segment):
                s = slice(i, i + segment)
                o, state = _orig(q[:, s], k[:, s], v[:, s], g=g[:, s], beta=beta[:, s], initial_state=state,
                                 output_final_state=True, **kw)
                outs.append(o)
            return torch.cat(outs, dim=1), (state if want_final else None)

        mod.torch_chunk_gated_delta_rule = seg_rule
        patched.append(name)
    return patched
