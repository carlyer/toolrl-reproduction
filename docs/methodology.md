# 训练方法与评测协议

## 背景

ToolRL使用细粒度工具调用reward训练模型。要证明本地训练有效，需要用同一基模和训练后checkpoint在完整benchmark上比较，而不是引用训练validation reward。

## 训练

- 基模：Qwen2.5-3B-Instruct。
- 官方rlla_4k：3920train /80validation。
- 15epochs，最终global_step104；学习率1e-6。
- train batch512、minibatch128、每题4候选、response1024。
- 本地4卡，官方发布脚本默认2卡；有效batch保持。
- prompt4608代替论文2048，以保留全部样本，属于明确协议差异。
- KL reward系数显式0，按论文说明而不是官方脚本原0.001；该差异需随结果披露。
- Torch2.4.0、vLLM0.6.3、Transformers4.47.1及官方veRL版本；最终checkpoint补丁单独保留。

## API-Bank

597条：Level1=399、Level2=67、Level3=131。使用作者processed prompt与soft-match，另报告只允许一个调用且完整等于first gold的strict诊断。

作者soft规则在某些无name/parameters外壳的输出上回退为参数匹配，也允许多预测中任一匹配；严格结果因此较低。`api_bank.py`明确实现两种口径，不能混报。

同两臂4096上下文/生成预算；batch离线推理替代作者逐条；请求temperature0.0001被vLLM0.6.3提升为实际0.01，seed42。两个值和批次差异均记录，不称完全确定性greedy。

## BFCL V3

固定Gorilla提交`ea13468e4423454d0c213704fb87cf7cb3990433`，17类共4441任务。ToolRL提示/解析适配到该版本，官方generator/checker保持。每请求独立Dialogue避免状态污染。

两臂32768上下文、4096生成上限，BF16，评测不截断历史。上下文耗尽属于模型预算失败，留在分母；连接/服务异常属于基础设施失败，不得产出“完整”结果。

作者未记录原始BFCL精确提交；当前冻结版本是V3，并非已确认与作者完全相同。各环境隔离，不把后续V4成绩当作论文V3结果。
