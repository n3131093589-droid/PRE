# DrugBank Dynamic Node Gating Experiments

## Goal

先验证一件事：同一个药物在不同 partner drug 条件下，是否需要不同的子结构权重。

当前分支新增了 `--node_gate_mode`：

- `0`: 关闭动态节点门控，保持原始层级传播。
- `1`: 打开 partner-aware 子结构门控，在所有 block 中对 `y == 1` 的子结构节点生效。
- `2`: 打开 relation-aware 子结构门控，在所有 block 中对 `y == 1` 的子结构节点生效。

门控是软门控，不改图拓扑，只对节点表示做条件缩放：

- 每个药物先汇总本药的子结构上下文。
- 对侧药物的上下文作为条件信号。
- 每个子结构节点得到一个按 pair 动态变化的缩放系数。
- `node_gate_mode = 2` 时，会额外拼接 relation embedding，让 gate 从 pair-aware 升级为 pair + relation aware。
- 当前版本会在每个 block 中都施加门控，让 partner-aware 信号贯穿整个层级表示学习过程。

## Recommended Next Comparison

在当前结果里，建议先固定 `ablation_mode = 2`，只比较：

- `ab2 ng1`: 当前最优的 partner-aware 动态节点门控
- `ab2 ng2`: relation-aware 动态节点门控

这样可以把新增收益归因到 relation context，而不是别的结构改动。

## Update Layer Extension

当前分支进一步新增了 `--update_mode`，用于控制 Update layer 是否采用“关系条件残差更新”：

- `0`: 保持原始的 concat + norm 更新。
- `1`: 打开 relation-aware residual update，在每个 block 中把 relation embedding 作为条件残差写回融合表示。
- `2`: 打开 relation-aware bidirectional residual update，先分别用 `intra + inter + e(r)` 和 `inter + intra + e(r)` 更新两路表示，再做原始融合。

这里不是 gate 版本，而是 residual 版本：

- 先得到原始融合表示 `concat(intra_rep, inter_rep)`。
- 再把 relation embedding 投影到 block hidden dim。
- 用一个小 MLP 生成条件残差并加回融合表示。
- 最后再做原有的 norm 和 pooling。

`update_mode = 2` 则更进一步：

- 不再只对融合后表示做一次 residual 修正。
- 而是先分别更新 `intra_rep` 和 `inter_rep`。
- `intra_rep` 的修正由 `intra_rep + inter_rep + e(r)` 共同决定。
- `inter_rep` 的修正由 `inter_rep + intra_rep + e(r)` 共同决定。
- 两路都完成关系条件残差更新后，再拼接进入原有的 `norm + pooling`。

建议实验时先固定之前各自最优的节点门控配置：

- warm-start: 先比较 `ab2 ng2 um0` 对 `ab2 ng2 um1`
- cold-start: 先比较 `ab2 ng1 um0` 对 `ab2 ng1 um1`

如果 `um1` 已经证明有效，下一步建议直接比较：

- warm-start: `ab2 ng2 um1` 对 `ab2 ng2 um2`
- cold-start: `ab2 ng1 um1` 对 `ab2 ng1 um2`

这样可以把收益尽量归因到 Update layer 本身，而不是重新混入节点门控因素。

## Checkpoint Naming

训练和测试脚本现在都会自动追加后缀：

- `-ab{ablation_mode}-ng{node_gate_mode}`

示例：

- `./pkl/db-subg-fold0-ab2-ng0.pkl`
- `./pkl/db-subg-fold0-ab2-ng1.pkl`
- `./pkl/db-subg-fold0-ab2-ng2-um1.pkl`

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