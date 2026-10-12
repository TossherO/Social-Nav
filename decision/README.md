# 决策域

## 局部场景决策：`vlm/`

现有JRDB输入、人工审核、可视化、远程测速和打包的唯一真实实现。

- 默认冻结Qwen3.8-27B-FP8、非思考、object/JSON Schema、每2秒一次。
- T1已有离线工具；T2/T3只有接口定义，不是已完成在线能力。
- 公开配置、Prompt、Schema在`vlm/configs`；敏感`configs/*.local.json`仍在根目录。
- `data`和`outputs`保持原位，既有输入/结果不改写。
- `deployment/`：AutoDL原环境安装/核验/手动服务工具，详见[部署与测试指南](vlm/deployment/README.md)。
    - 直接替换远程容器原有Python环境的依赖为Torch 2.13+cu130、SGLang 0.5.19，不新建服务虚拟环境。
    - `install`/`serve`默认只打印；安装须`--execute --allow-env-change --out`明确确认并记录。CUDA13远程安装/推理尚未验证，不修改本机开发环境。
    - `download`仅准备wheel及间接依赖，可在无GPU机器上执行；`install --wheelhouse`校验完整锁文件后离线安装，不退回在线下载。第三方依赖ZIP与轻量测试包分开。
    - `--with-deployment`便携包复用同一benchmark和24例冻结资产，不含权重或凭证；客户端流程见[便携测试包说明](vlm/configs/vlm_test_guide.md)。

入口示例：

```bash
python -m decision.vlm.benchmark_vlm --dry-run --limit 24
python -m decision.vlm.jrdb_t1 prepare --scenes clark --frames 75 --out outputs/t1/new_case
```

scripts中的旧入口仅转发main，仍支持python -m scripts.<原命令>；
库函数请从decision.vlm导入。便携包仍是独立benchmark.py，不需要完整系统。

## 外层长程决策：`agent/`

未来长程外层决策位置，没有选定/安装agent框架。

- 预期承担任务分解、检查点/模式切换、调用VLM或导航技能、恢复与任务记忆。
- 不替代0.5Hz局部社会语义，不直接发底盘速度，不解除本地安全停车。
- 在真实反馈、技能边界和测试建立前不做自动tool调用或假成功循环。

未来的意图/观察/导航反馈草案见[共享类型](../interfaces/contracts.py)；尚未连接当前VLM HTTP。
