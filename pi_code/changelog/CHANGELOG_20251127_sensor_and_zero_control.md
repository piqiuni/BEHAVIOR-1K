# 2025 年 11 月 27 日 - 传感器修改与0位控制（pi_dev 分支）

### 1. 新增（Added）

* 新增键盘控制策略服务器 `keyboard_policy_server.py`，支持通过键盘实时控制 23 自由度机器人关节
  - 支持关节选择（按键 1/2）
  - 支持关节增减量控制（按键 [/]，步长 0.1）
  - 支持关节归零重置（按键 r）
  - 优化终端显示，实时显示当前控制状态（关节索引、关节值、步数）
  - 使用独立线程处理键盘输入，避免阻塞 WebSocket 通信

### 2. 修复（Fixed）

* 无

### 3. 优化（Changed）

* 优化 `behavior_env_web.py` 环境客户端
  - 添加 RichObservationWrapper 封装，提供更丰富的观测数据
  - 修改主循环逻辑，移除 for 循环改用 while 循环控制
  - 注释掉自动终止逻辑，添加手动按 Enter 继续下一轮的交互方式
  - 预留传感器可视化接口（注释状态）
* 优化 `simple_websocket_server.py` 测试服务器
  - 将测试动作从零值改为 -1 数组，便于验证动作是否生效
* 优化 `robot_control_example.py` 示例代码
  - 添加调试注释，预留动作打印接口

### 4. 移除（Removed）

* 无

### 5. 技术细节（Technical Details）

**键盘控制策略服务器特性：**
- 动作维度：23（R1Pro 机器人标准）
- 动作持久化：关节状态在修改前保持不变
- 非阻塞输入：使用 termios/tty/select 实现单字符非阻塞读取
- 日志优化：websockets 日志级别调整为 WARNING，减少控制台干扰
- 状态显示：格式 `[CTRL] Joint: XX | Value: ±X.XXX | Steps: XXXX`

**环境客户端增强：**
- 观测封装器：RichObservationWrapper 提供额外观测模态
- 控制流程：手动推进 episode，方便调试和数据采集
- 可扩展性：预留传感器可视化钩子（robot.visualize_sensors）

**使用示例：**
```bash
# 启动键盘控制服务器
python keyboard_policy_server.py

# 启动环境客户端（另一终端）
python -m omnigibson.examples.environments.behavior_env_web

# 控制说明
# 1/2: 切换控制的关节索引
# [/]: 减小/增大当前关节值
# r: 重置所有关节归零
# h: 显示帮助信息
# q: 退出服务器
```

**关键代码变更：**
- `behavior_env_web.py`: 添加 `RichObservationWrapper`，修改主循环为 `while step != max_steps`
- `keyboard_policy_server.py`: 新增完整的键盘控制策略实现
- `simple_websocket_server.py`: 测试动作改为 `[-1] * action_dim` 数组
