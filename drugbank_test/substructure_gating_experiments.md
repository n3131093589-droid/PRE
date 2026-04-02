# DrugBank Dynamic Node Gating Experiments

## Goal

先验证一件事：同一个药物在不同 partner drug 条件下，是否需要不同的子结构权重。

当前分支新增了 `--node_gate_mode`：

- `0`: 关闭动态节点门控，保持原始层级传播。
- `1`: 打开 partner-aware 子结构门控，在所有 block 中对 `y == 1` 的子结构节点生效。

门控是软门控，不改图拓扑，只对节点表示做条件缩放：

- 每个药物先汇总本药的子结构上下文。
- 对侧药物的上下文作为条件信号。
- 每个子结构节点得到一个按 pair 动态变化的缩放系数。
- 当前版本会在每个 block 中都施加门控，让 partner-aware 信号贯穿整个层级表示学习过程。

## Checkpoint Naming

训练和测试脚本现在都会自动追加后缀：

- `-ab{ablation_mode}-ng{node_gate_mode}`

示例：

- `./pkl/db-subg-fold0-ab2-ng0.pkl`
- `./pkl/db-subg-fold0-ab2-ng1.pkl`

## Primary Matrix

先跑主对照，判断这个方向值不值得继续。

| ID | Split | Purpose | Command Delta |
| --- | --- | --- | --- |
| C0 | cold-start | 当前增强版基线 | `--ablation_mode 2 --node_gate_mode 0` |
| C1 | cold-start | 当前增强版 + 动态节点门控 | `--ablation_mode 2 --node_gate_mode 1` |
| W0 | warm-start | 当前增强版基线 | `--ablation_mode 2 --node_gate_mode 0` |
| W1 | warm-start | 当前增强版 + 动态节点门控 | `--ablation_mode 2 --node_gate_mode 1` |

优先比较：

- cold-start: `C1` 对 `C0`
- warm-start: `W1` 对 `W0`

## Attribution Matrix

如果主对照有提升或基本持平，再跑归因对照，确认收益是不是来自节点门控本身，而不是和虚拟交互支路偶然耦合。

| ID | Split | Purpose | Command Delta |
| --- | --- | --- | --- |
| C2 | cold-start | 原始交互基线 | `--ablation_mode 1 --node_gate_mode 0` |
| C3 | cold-start | 原始交互 + 动态节点门控 | `--ablation_mode 1 --node_gate_mode 1` |
| W2 | warm-start | 原始交互基线 | `--ablation_mode 1 --node_gate_mode 0` |
| W3 | warm-start | 原始交互 + 动态节点门控 | `--ablation_mode 1 --node_gate_mode 1` |

## Recommended Run Order

1. 先跑 `C0` 和 `C1`，看 cold-start 是否受益。
2. 再跑 `W0` 和 `W1`，看 warm-start 是否一致。
3. 如果两组里至少一组持平或提升，再补 `C2/C3` 和 `W2/W3`。

## Command Templates

### Cold-start

```bash
python drugbank_test/inductive_train.py \
  --fold 0 \
  --batch_size 512 \
  --n_atom_feats 66 \
  --device 0 \
  --ablation_mode 2 \
  --node_gate_mode 0 \
  --pkl_name ./pkl/db-subg.pkl
```

把 `--node_gate_mode 0` 改成 `1` 就是动态节点门控版本。

### Warm-start

```bash
python drugbank_test/transductive_train.py \
  --fold 0 \
  --batch_size 1024 \
  --device 0 \
  --ablation_mode 2 \
  --node_gate_mode 0 \
  --pkl_name ./pkl/db-subg.pkl
```

## Acceptance Rule

建议按下面的顺序判断是否继续推进到“动态子结构选择”。

1. 如果 `C1` 和 `W1` 至少有一组稳定提升，另一组不明显退化，这个方向可以继续。
2. 如果两组都基本持平，也可以继续，但优先先做门控可视化，再决定要不要上硬选择。
3. 如果两组都明显下降，就不要直接上“动态子结构选择”，先回头检查门控强度、初始化和条件摘要方式。

## Result Table Template

| ID | Fold | ACC | AUROC | F1 | AUPR/AP | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 0 |  |  |  |  |  |
| C1 | 0 |  |  |  |  |  |
| W0 | 0 |  |  |  |  |  |
| W1 | 0 |  |  |  |  |  |

## Next Step Decision

只有当动态节点门控版本在主对照里表现可接受时，才建议进入下一步：

- 从软门控升级到 top-k 动态子结构选择。
- 或者先把门控权重导出来，确认不同 partner 下激活的子结构确实不同。