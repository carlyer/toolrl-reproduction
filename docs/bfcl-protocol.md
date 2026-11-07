# BFCL V3 接入

官方仓库 tag v1.3 / ea13468e4423454d0c213704fb87cf7cb3990433，17类4441条。该版本数据和计分仍为BFCL_v3；不使用RC-GRPO目录的V4版本。原ToolRL未记录其精确BFCL提交，因此此冻结版本不是已确认的原作者同一提交。

ToolRL官方rlla_qwen.py完整提示/解析保留，bfcl命名空间改为bfcl_eval；新API取消turn_type参数，因此使用每请求独立Dialogue对象携带single/multi属性，避免多线程状态污染。官方generator与checker不替换。两臂相同temperature请求0.0001（legacy vLLM实际0.01），bf16，上下文32768。本地独立评测环境/tmp/toolrl-bfcl-v3-venv，vLLM服务仍使用已验证的官方旧版训练环境；不改变全局包。

运行入口scripts/evaluation/bfcl_pipeline.py；每臂必须完成4441唯一ID才进入官方checker。失败写状态，不宣称完整指标；服务器使用独立进程组，阶段退出清理其子进程。只GPU0单卡持锁。
