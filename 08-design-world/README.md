# 08 · DesignWorld

**Can computer-use agents rebuild real designs in real design software?**

DesignWorld is an open project with [Perpetual](https://perpetual.video). It has five parts:

| Part | What it is |
|---|---|
| **DesignGym** (`envs/designgym`) | An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment. An agent sees a target design and rebuilds it with tool calls, with HTML, or through a GUI: six editors modelled on common design tools, plus real Inkscape, LibreOffice Impress and LibreOffice Draw. Scored on submit against the design's layers. |
| **DesignWorldBench** (`bench/`) | The benchmark. v1, *Recreation-CUA*, is a frozen eval set with a computer-use harness. Every eval run is published as a CUA dataset. |
| **DesignWorld-SFT** | 2k designs, each recorded in 14 GUI editors, plus tool-call and HTML versions: 16 subsets. |
| **Training** (`training/`) | SFT with TRL; RL with Tinker, SkyRL or Miles. Main models: Qwen3.6-35B-A3B and Qwen3.8-27B. |
| **World model** (`world-model/`) | An image model fine-tuned to predict the next screen after a GUI action. |

DesignGym ships 4k tasks: a 2k `sft` split, a 2k `rl` split and the frozen `test` split used by DesignWorldBench.

> This folder is being set up. Code moves in from the earlier `07-design-recreation` work.
