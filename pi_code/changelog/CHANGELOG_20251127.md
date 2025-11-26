# 更新日志

## 2025年11月27日 - WebSocket通信基础设施

**Commit**: 修改web policy 目前可接收与发送
**分支**: pi_dev

---

### 新增功能

#### WebSocket服务器实现 (`simple_websocket_server.py`)

- 为msgpack协议添加了自定义NumPy数组序列化支持（与OmniGibson的network_utils兼容）
- 添加健康检查端点（`/healthz`）用于连接状态监控
- 连接初始化时的元数据交换机制
- 可通过命令行参数配置服务器（主机、端口、动作维度）
- 改进了消息处理的错误处理和日志记录
- 添加接收观测数据的调试消息跟踪

#### WebSocket环境演示 (`behavior_env_web.py`)

- 新增通过WebSocket控制BEHAVIOR机器人的示例脚本
- 观测数据预处理流程（匹配eval.py格式）
- 多摄像头设置的相机相对位姿计算
- 任务ID编码支持
- 命令行界面支持任务选择和服务器配置

### 问题修复

#### WebsocketPolicy动作转换 (`policies.py`)

- 修复了 `'numpy.ndarray' object has no attribute 'detach'`错误
- 添加类型检查以处理Tensor和NumPy数组两种返回类型
- 正确将来自WebSocket服务器的NumPy动作转换为PyTorch张量

#### 数组张量转换 (`array_tensor_utils.py`)

- 增强 `torch_to_numpy()`函数以处理混合类型（Tensor、NumPy、list、tuple）
- 防止数据结构中已存在NumPy数组时的转换错误
- 统一dtype处理（转换为float32）以避免下游类型不匹配

### 配置变更

#### 机器人配置 (`r1pro_behavior.yaml`)

- 将机器人名称从 `r1prorobot_r1robot_r1`修复为 `robot_r1`以保持一致性

### 代码质量改进

- 为文档字符串添加了空值安全检查
- 注释掉调试用的print语句
- 改进了所有组件的日志消息

---

### 技术细节

- **消息协议**: msgpack + 自定义NumPy数组序列化/反序列化
- **默认配置**: localhost:8000，23维动作空间（R1Pro机器人）
- **观测格式**: 扁平化字典，包含RGB/深度图像、本体感受、相机位姿、任务ID
- **动作格式**: NumPy数组形状为(23,)，用于R1Pro机器人控制

### 文件变更摘要

| 文件                           | 变更类型  | 说明                                 |
| ------------------------------ | --------- | ------------------------------------ |
| `simple_websocket_server.py` | 新增/修改 | WebSocket服务器实现，支持NumPy序列化 |
| `behavior_env_web.py`        | 新增/修改 | WebSocket控制的环境演示脚本          |
| `policies.py`                | 修复      | 修复动作类型转换错误                 |
| `array_tensor_utils.py`      | 增强      | 改进混合类型数据转换                 |
| `r1pro_behavior.yaml`        | 修复      | 修正机器人名称配置                   |

---

### 使用方法

#### 启动WebSocket服务器

```bash
# 默认端口（8000）
python simple_websocket_server.py

# 自定义端口
python simple_websocket_server.py --port 9000

# 自定义主机和端口
python simple_websocket_server.py --host 0.0.0.0 --port 8080
```

#### 启动环境客户端

```bash
# 默认设置
python -m omnigibson.examples.environments.behavior_env_web

# 自定义服务器
python -m omnigibson.examples.environments.behavior_env_web --host localhost --port 8000

# 不同任务
python -m omnigibson.examples.environments.behavior_env_web --task cleaning_bathrooms

# 无头模式
python -m omnigibson.examples.environments.behavior_env_web --headless
```

---

**贡献者**: piqiuni@qq.com
