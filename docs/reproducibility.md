# 复现指南

## CPU

```bash
python examples/demo.py
make check
pip install -r requirements-viz.txt
make figures
```

## 旧训练环境

在支持Torch2.4/cu121、vLLM0.6.3的GPU机器上创建单独Python3.11环境。不能用系统CUDA13直接编译后声称兼容旧FlashAttention ABI。

```bash
bash scripts/bootstrap_upstream.sh
python3.11 -m venv .venv
.venv/bin/pip install -r configs/requirements.txt
# 安装匹配Torch/CUDA/ABI的FlashAttention2.7.2.post1
.venv/bin/pip install -e upstream --no-deps
export BASE_MODEL="$PWD/models/Qwen2.5-3B-Instruct"
.venv/bin/python scripts/verify.py
CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 bash scripts/train.sh smoke
CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 bash scripts/train.sh full
```

`bootstrap_upstream.sh`固定官方提交并应用本地兼容及final-checkpoint补丁；不拉取新的默认分支作为复现基准。

## API-Bank

```bash
.venv/bin/python scripts/evaluation/api_bank.py --model "$BASE_MODEL" --output results/runs/api-base
.venv/bin/python scripts/evaluation/api_bank.py --model "$GRPO_MODEL" --output results/runs/api-grpo
```

模型、输入和官方processed benchmark必须就绪。输出可断点恢复，但已有预测必须属于同一个模型；不要复用不同checkpoint的目录。

## BFCL V3

BFCL官方checker在独立`.venv-bfcl`安装固定Gorilla提交`ea13468e4423454d0c213704fb87cf7cb3990433`的`berkeley-function-call-leaderboard`目录。按`notes/bfcl-v3-integration.patch`在该固定工作树加入ToolRL提示/解析和上下文预算处理；补丁路径相对Gorilla仓库根目录。

```bash
git clone --filter=blob:none https://github.com/ShishirPatil/gorilla.git gorilla
git -C gorilla checkout ea13468e4423454d0c213704fb87cf7cb3990433
python scripts/install_bfcl_adapter.py --gorilla gorilla
python3.11 -m venv .venv-bfcl
.venv-bfcl/bin/pip install -e ./gorilla/berkeley-function-call-leaderboard
export GRPO_MODEL="$PWD/results/checkpoints/<run>/actor/global_step_104"
export BFCL_PYTHON="$PWD/.venv-bfcl/bin/python"
export TOOLRL_PYTHON="$PWD/.venv/bin/python"
TOOLRL_EVAL_GPU=0 python scripts/evaluation/bfcl_pipeline_fixed.py
```

训练与checker依赖相互隔离。流水线只清理自己创建的vLLM进程组；不得中断其他训练。新结果使用新目录并保存实际环境版本。原始模型权重、预测和raw数据不随仓库分发。

## Evidence and release boundaries

The reported GPU experiments were run in October 2026. This public export adds
portable paths, CPU demos, documentation, charts and validation workflows. Those
packaging changes are not a new GPU replication.

`results/source-manifest.json` records SHA256 hashes of the selected original
workspace files and their initial sanitized exports. `publication-manifest.json`
records the final public source, evidence and figure hashes. These hashes support
artifact traceability; they do not independently verify that a GPU run happened.

Only code, configuration and aggregate evidence are included. Model weights,
raw datasets, full prompts and raw model transcripts are obtained separately.
Exact historical replay requires the original frozen inputs and checkpoint
artifacts; a new dataset or checkpoint creates a new experiment. Existing test
cohorts have been observed and must not be used as new blind tests.

CPU checks validate scoring, guards and released aggregates. They do not certify
GPU dependency compatibility or reproduce training accuracy. Follow the recorded
environment and use a new run directory for GPU experiments. Final-only saves
avoid redundant base and optimizer copies.

`notes/local-adaptations.patch`已经包含final-checkpoint修改；`final-checkpoint.patch`仅作为该子改动的独立审阅材料，不重复应用。BFCL handler源自官方ToolRL，命名空间与每请求Dialogue是本地适配，见`adapters/bfcl/rlla_qwen.py`。
