
## [WebSocket 基础设施] - 2025-11-27 (`pi_dev`)

### ✨ 新增 (Added)

* **核心服务实现** ：支持 msgpack + NumPy 序列化，适配 OmniGibson 机器人控制场景。
* *涉及文件：* `simple_websocket_server.py` (Server), `behavior_env_web.py` (Client)
* **CLI 配置增强** ：支持自定义主机、端口、任务选择及无头模式 (Headless) 启动参数。

### 🐛 修复 (Fixed)

* 修复 `numpy.ndarray` 缺失 `detach` 属性导致的类型转换崩溃问题。
* 修复机器人配置名称匹配逻辑，确保配置加载准确性。

### ♻️ 优化 (Changed)

* **日志与代码质量** ：重构日志格式以提升排查效率；清理冗余逻辑与补充注释。
* **数据处理适配** ：
* `policies.py`：适配 WebSocket 通信的策略逻辑。
* `array_tensor_utils.py`：优化 Tensor/Array 转换工具链。

---

## [传感器修改与Robot 0位控制] - 2025-11-27 (`pi_dev`)

### ✨ 新增 (Added)

* **键盘控制服务** (`keyboard_policy_server.py`)：
  * 实现 23 自由度机器人实时控制（独立线程处理，不阻塞通信）。
  * *交互功能：* 关节选择 (`1`/`2`)、步长微调 (`[`/`]`, step=0.1)、归零重置 (`r`)。
  * *界面优化：* 终端实时显示当前关节索引、数值及步数。

### ♻️ 优化 (Changed)

* **环境客户端** (`behavior_env_web.py`)：
  * 集成 `RichObservationWrapper` 提供丰富观测数据。
  * 逻辑重构：改为 `while` 循环控制，支持按 `Enter` 手动步进，预留可视化接口。
* **测试与示例** ：
* `simple_websocket_server.py`：测试动作基准值调整为 -1 数组（验证动作生效）。
* `robot_control_example.py`：增加调试注释与动作打印接口。
