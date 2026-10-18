# FSDB validated mapping extraction

本目录只整理历史 ImageNet R34/R50 frozen-mag QAT 中已经验证的两种分组：

- **4:1 `e_state_oc`**：每个输出通道（OC）独立；先按 1:1 最佳状态 `state=2*sign+shift`，再按 `mag`、整数 `target` 和原逻辑索引稳定排序，每连续 4 个权重组成一组。
- **16:1 `h_error_vector_oc`**：每个 OC 独立；对每个权重计算四种 `(sign,shift)` 的整数域平方误差，按 `best_state,e0,e1,e2,e3,target` 和原逻辑索引进行字典序排序，每连续 16 个权重组成一组。

这两者都是数据依赖的静态 permutation mapping。QAT 中 `group_members`、mag、weight/input scale 固定；每次 forward 根据当前 shadow weights 为每组重新选择共享 sign/shift，然后用 identity STE 把 hard-weight 梯度传回 shadow weights。

## 历史结果

| 网络 | 4:1 `e_state_oc` Top-1 | 16:1 `h_error_vector_oc` Top-1 |
|---|---:|---:|
| R34 | 73.226% | 73.248% |
| R50 | 75.554% | 75.260% |

以上是 10 epoch 后 official ImageNet-1K validation 的 final-epoch Top-1；详细 hash 和 Top-5 见 `configs/`。

## 代码对应关系

- `src/fsdb_mapping_repro/mapping_methods.py`：从历史 `run_r18_ptq_sharing_mapping_search.py` 的 `partitions`、`feature_keys`、`sorted_partitions`、`error_signature`、`candidate_partitions`、`make_groups`，以及 scaling runner 的 `build_groups` 原样抽取；候选表只保留上述两项。
- `src/fsdb_backend/quantizer.py`：权重字段编码/解码；历史源文件原样复制。
- `src/fsdb_backend/qat_*.py`：shadow weight、STE、frozen mag、固定 mapping gather/decode/scatter；历史源文件原样复制。
- `configs/r34_imagenet.json`、`configs/r50_imagenet.json`：从历史 `protocol.json` 精简出的真实训练协议、源 checkpoint、结果与 mapping hash。
- `tools/build_group_members.py`：只调用抽取出的原分组函数，从预先准备的逐层状态字典生成 group membership；它不训练、不评估。

未包含 Exact-MinRoute、其他 mapping、Vivado RTL、ImageNet、checkpoint、日志或预生成的大型 group tensor。

## 解码公式核查

历史实验实际调用 `fsdb_backend.quantizer.fsdb_decode`：

```text
shift=0: q = mag - 4*sign
shift=1: q = 8*(mag - 8*sign)
```

因此低位分支是 **M−4**（当 `sign=1`），不是 M−5。原位置：`/data/wangjr/DQAT/pytorch-imagenet/fsdb_backend/quantizer.py:28-41`；本目录相同函数位于 `src/fsdb_backend/quantizer.py:28-41`。mapping 搜索 LUT 由该函数在历史 `run_r18_ptq_sharing_mapping_search.py:406-424` 直接生成；QAT decode 在历史 `fsdb_backend/qat_mapping_frozen_mag.py` 的 `project_mapping_fixed_mag` 中调用同一函数。

## 依赖与使用

最低 Python 依赖见 `requirements.txt`。仅生成 mapping 时，输入 `.pt` 必须包含：

```text
selected_layers: [layer_name, ...]
layers[layer_name]: shape, weight, target, q1, mag, sign, shift, scale, zero
```

`weight/target/q1/mag/sign/shift` 应为 NumPy 数组，且必须来自同一个完成后的 1:1 frozen-mag checkpoint；不能重新校准后混用。
该 `.pt` 是 pickle-backed 输入，只应加载由自己从可信 checkpoint 生成的文件。

```bash
export PYTHONPATH=$PWD/src
python tools/build_group_members.py \
  --layer-state /path/to/layer_state.pt \
  --method e_state_oc \
  --output /tmp/r4_e_state_oc_group_members.pt

python tools/build_group_members.py \
  --layer-state /path/to/layer_state.pt \
  --method h_error_vector_oc \
  --output /tmp/r16_h_error_vector_oc_group_members.pt
```

历史训练的真实入口仍是原仓库 `run_fsdb_frozen_mag_scaling_imagenet.py:911-931`：

```bash
python run_fsdb_frozen_mag_scaling_imagenet.py prepare --arch r34
python run_fsdb_frozen_mag_scaling_imagenet.py baseline --arch r34
python run_fsdb_frozen_mag_scaling_imagenet.py preflight --arch r34
python run_fsdb_frozen_mag_scaling_imagenet.py branch --arch r34 --branch r4_e_state_oc
python run_fsdb_frozen_mag_scaling_imagenet.py branch --arch r34 --branch r16_h_error_vector_oc
```

R50 将 `r34` 改为 `r50`。这些命令仅用于记录历史入口，本整理过程未执行它们。

## 当前复现边界

本目录足以静态审查解码、两种分组和 mapping-aware frozen-mag QAT 投影，也能在提供上述逐层状态字典时重建分组。它**不是开箱即用的端到端 ImageNet 训练仓库**：历史 runner 依赖原仓库的数据划分、校准/1:1 baseline 流程、模型构造与训练工具；为避免复制无关实验代码，这些未纳入。

缺失的大型/外部项如下：

- ImageNet-1K 数据及授权路径；
- R34/R50 源 checkpoint（路径和 SHA256 已写入配置）；
- R34/R50 完成后的 1:1 baseline checkpoint与逐层状态导出；
- calibrated quant params（R34 历史来源写入配置；R50 由 prepare 生成）；
- 约 165–169 MB/个的历史 `group_members.pt`（原路径与验证 hash 已写入配置，未复制）；
- 原 runner 所需 `timm/torchvision`、数据加载及训练 orchestration。

因此，没有这些项时只能检查/单元测试算法，不能声称复现历史 Top-1。
