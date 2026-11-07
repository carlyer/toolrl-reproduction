# 双基准结果与局限

API-Bank author soft：33.33%→54.27%（+20.94pp）；strict单调用：32.33%→52.26%。1194条预测完成作者评分逻辑对齐。

BFCL V3官方Overall：39.87%→53.97%（+14.10pp）。Non-live AST提升明显（45.10→83.17），Multi-turn仍仅1.62→5.25%。不要用整体得分掩盖多轮能力不足。

## 完整分母与失败

- 每臂API-Bank597条、BFCL4441唯一ID。
- Base55条、GRPO25条上下文预算失败，全部由checker作为任务失败计入。
- BFCL基础设施错误0；审计器对不完整ID集合或非预算推理错误直接抛错。
- 未完成Bamboogle，因为需要额外搜索服务；没有填补或估计成绩。

## 与论文结果的差异

参考论文目标API-Bank67.00%、BFCL V3 52.98%、Bamboogle60.00%。本地API-Bank未达到目标，BFCL数值更高也不证明整体优于原论文：精确BFCL版本、训练prompt预算、KL配置和生成温度均有差异。

一个训练seed及固定checkpoint的比较不提供训练方差结论；当前结果是本地复现工程证据，不是新算法或全文指标认证。

## 证据

[benchmarks.json](../results/benchmarks.json)保存官方Overall与各子项，API-Bank分数、分母和错误政策。`python scripts/report.py --check`重读公开聚合；完整原预测与权重需另取得，不能由这份聚合重建每条预测。
