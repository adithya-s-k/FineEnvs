# World model

**Status: there is no code yet.** This page describes the plan.

The goal is an image model that predicts the next screen of a design editor. It takes two inputs: the current
screenshot and the GUI action the agent takes.

```
(screenshot_t, action_t)  ->  screenshot_t+1
```

The actions are DesignGym's GUI tools: `left_click`, `left_click_drag`, `type`, `key`, `scroll` and the others,
with their JSON arguments. A model that renders the editor's response well could stand in for the editor. That
makes it useful for planning, data augmentation and RL in places where running the real app is slow.

## Data

The training pairs are `(screenshot_t, action_t, screenshot_t+1)`, cut from GUI trajectories in the `cua-v1`
format. There are two sources:

- **DesignWorld-SFT, GUI subsets.** Verified trajectories that rebuild the same designs in many editors.
- **DesignWorldBench runs** ([`FineEnvs/designworld-bench-runs`](https://huggingface.co/datasets/FineEnvs/designworld-bench-runs)).
  These are model-driven trajectories, mistakes included, in the built-in editors and in the real apps
  (Inkscape, LibreOffice Impress, LibreOffice Draw).

## Extracting pairs from cua-v1 rows

A row's `images` list follows the order of its `{"type": "image"}` parts in `messages`, so the two can be walked
together:

1. **Track the current screen.** Walk the messages in order and count image parts to index into `images`. In the
   opening user turn, the target image comes first in reference mode, then the first screenshot. That first
   screenshot is the current screen.
2. **Read the actions.** Each assistant message holds one or more `tool_calls`. Each call's result is a `tool`
   message with the same `tool_call_id`. On the GUI surface, the screenshot taken after a turn sits in that
   turn's last tool message.
3. **Emit a pair.** When an assistant turn has exactly one GUI action, emit (the current screen, that call, the
   next screenshot). Then make that screenshot the current screen.
4. **Handle multi-action turns.** A turn with several actions makes one multi-step pair, or is skipped.
   Calls to `screenshot`, `get_task` and `view_reference` change nothing on screen, so drop them from the action.
5. **Normalise coordinates.** Use `screen` (the screenshot size) and `coordinates` (`pixels`, or `normalized`
   on a 0-1000 grid). Encode the action as text, e.g. the tool call as JSON, so any image-editing model can take
   it as an instruction.

Bench rows add `error`, `reward` and `missing_images`. Skip pairs from episodes that are missing screenshots.

## Candidate models

The plan is to fine-tune an open image-editing model: one that takes an input image plus a text instruction and
outputs an edited image. Here the action text is the instruction. The other route is an action-conditioned
video or next-frame model that starts from the same kind of backbone. To be chosen.

## Evaluation ideas

- **Pixel fidelity.** PSNR, SSIM and LPIPS against the true next frame, reported separately for frames that
  change a lot (dialogs, new elements) and for frames that barely change.
- **Action-conditioned accuracy.** Did the predicted frame change in the right place? For example, the selection
  box appears at the clicked element, or typed text appears in the focused field. This compares against a
  no-op baseline (copy screenshot_t) and a random-action baseline.
- **Rollouts.** Run open-loop for several steps and measure drift: how many steps until the predicted state
  departs from the real one.
- **As a simulator.** Train or evaluate an agent inside the model, then replay its actions in the real editor.
  Check whether the rewards and the agent's ranking carry over to DesignGym.
