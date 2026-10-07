决策域（2026-10-07）

vlm/：现有JRDB输入、人工审核、可视化、远程测速和打包的唯一真实实现。
  默认冻结Qwen3.8-27B-FP8、非思考、object/JSON Schema、每2秒一次。
  T1已有离线工具；T2/T3只有接口定义，不是已完成在线能力。
  公开配置、Prompt、Schema在vlm/configs；敏感configs/*.local.json仍在根目录。
  data和outputs保持原位，既有输入/结果不改写。

入口示例：
  python -m decision.vlm.benchmark_vlm --dry-run --limit 24
  python -m decision.vlm.jrdb_t1 prepare --scenes clark --frames 75 --out outputs/t1/new_case

scripts中的旧入口仅转发main，仍支持python -m scripts.<原命令>；
库函数请从decision.vlm导入。便携包仍是独立benchmark.py，不需要完整系统。

agent/：未来长程外层决策位置，没有选定/安装agent框架。
  预期承担任务分解、检查点/模式切换、调用VLM或导航技能、恢复与任务记忆。
  不替代0.5Hz局部社会语义，不直接发底盘速度，不解除本地安全停车。
  在真实反馈、技能边界和测试建立前不做自动tool调用或假成功循环。

未来的意图/观察/导航反馈草案见interfaces/contracts.py；尚未连接当前VLM HTTP。
