# Twosides Extension Experiments

## Goal

把已经在 DrugBank 上验证过的三类改动同步迁移到 Twosides：

- V-enhanced interaction branch
- substructure extraction with dynamic node gating
- update layer with relation-aware residual update

当前 Twosides 主线已经包含 V 分支，新增实验开关主要集中在后两项。

## New Flags

### node_gate_mode

- `0`: 关闭动态节点门控，保持原始层级传播。
- `1`: 打开 partner-aware 子结构节点门控，只对 `y == 1` 的子结构节点生效。
- `2`: 打开 relation-aware 子结构节点门控，在 partner-aware 基础上额外拼接 relation embedding。

### update_mode

- `0`: 保持原始 concat + norm 更新。
- `1`: 打开 relation-aware residual update，在每个 block 中对融合表示加入关系条件残差。
- `2`: 打开 relation-aware bidirectional residual update，先分别更新 intra 和 inter 两路表示，再进入原始融合。

这里的 update 不是 gate，而是 residual：

- 先得到 `concat(intra_rep, inter_rep)`。
- 再把 relation embedding 投影到当前 block hidden dim。
- 用小 MLP 生成条件残差并加回融合表示。
- 最后进入原有 norm 和 pooling。

`update_mode = 2` 会更进一步：

- 不再只对融合后的表示做一次残差修正。
- 而是先分别更新 `intra_rep` 和 `inter_rep`。
- `intra_rep` 的修正由 `intra_rep + inter_rep + e(r)` 决定。
- `inter_rep` 的修正由 `inter_rep + intra_rep + e(r)` 决定。
- 两路更新完成后，再拼接进入原有的 `norm + pooling`。

## Checkpoint Naming

Twosides 保持和旧实验兼容：

- 默认配置仍然是 `-ab{ablation_mode}`。
- 只有在打开 `node_gate_mode` 或 `update_mode` 时，才额外追加：
  - `-ng{node_gate_mode}`
  - `-um{update_mode}`

示例：

- `./pkl/ts-exp-fold0-ab2.pkl`
- `./pkl/ts-exp-fold0-ab2-ng1.pkl`
- `./pkl/ts-exp-fold0-ab2-ng1-um1.pkl`

## Recommended Validation Order

建议先固定 `ablation_mode = 2`，按最小增量验证：

1. `ab2` 对 `ab2 ng1`
2. `ab2 ng1` 对 `ab2 ng2`
3. `ab2 ng1` 对 `ab2 ng1 um1`
4. `ab2 ng1 um1` 对 `ab2 ng1 um2`

如果想先最省实验数，优先跑：

1. `ab2 ng1`
2. `ab2 ng2`
3. `ab2 ng1 um1`
4. `ab2 ng1 um2`

这样可以分别判断：

- partner-aware 子结构门控是否有效
- relation-aware 门控是否比 partner-aware 更强
- relation-aware residual update 是否能继续带来增益
- 双向关系残差更新是否能继续超过 um1

## Command Templates

### Train

```bash
python twosides_test/train.py \
  --fold 0 \
  --device 0 \
  --ablation_mode 2 \
  --node_gate_mode 1 \
  --update_mode 2 \
  --pkl_name ./pkl/ts-exp.pkl
```

### Test

```bash
python twosides_test/test.py \
  --fold 0 \
  --device 0 \
  --ablation_mode 2 \
  --node_gate_mode 1 \
  --update_mode 2 \
  --pkl_name ./pkl/ts-exp.pkl
```