# ndtimeline 设计文档

> **位置说明**：ndtimeline 模块位于 `legacy/vescale/ndtimeline/`，属于旧版 veScale（已整体迁入 `legacy/`，当前仓库根目录的活跃代码只有新版 DTensor）。本文档基于 legacy 代码分析。模块自带的简短 README 见 `legacy/vescale/ndtimeline/README.md`。

---

## 1. Motivation（设计动机）

在千卡乃至万卡规模上训练 LLM 时，性能诊断面临三个传统工具解决不了的问题：

1. **Straggler（掉队机器）难以定位**：个别机器发起 CUDA kernel 的速度变慢（"sink machines"），会拖慢整个集合通信，显著降低训练效率。要发现这类问题，必须**跨机器**对比同一时间窗口内各 rank 的行为。
2. **传统工具只能看单机**：torch profiler / nsys 只采集单进程或单机内的信息，而训练的性能瓶颈往往发生在成百上千台机器之间的通信上，单机视图无法呈现全局画面。
3. **全量 profiling 开销不可接受**：torch profiler 虽然细节丰富，但常开代价太高，且产生的 trace 文件巨大、难以分析，不适合生产环境长期开启。

因此需要一个满足以下目标的机制：

- **跨机器联合采集**：所有 rank 的 span 能放在同一个时间坐标系下对比；
- **低开销、可常开**：采集路径对训练关键路径的影响要足够小；
- **trace 文件小且易分析**：只记录感兴趣的粗粒度 span（前反向计算、P2P 收发、集合通信），而不是全量 kernel trace；
- **与并行语义一致、易扩展**：采集器位于每个训练进程内，天然继承每个 rank 的并行拓扑语义（TP/DP/PP/DDP），新增并行方式时无需改动采集核心。

## 2. Problem Definition（问题定义）

ndtimeline 要解决的核心问题可以形式化为：

> 在一个由 N 个进程（分布在多台物理机上）、以 TP × DP × PP（× DDP）方式组织的分布式训练任务中，以可常开的低开销，为每个 rank 上选定的操作记录形如 `(metric_name, absolute_start_ts, duration, tags, step)` 的 span，并使得**不同机器上记录的 span 可以直接放到同一个时间轴上对齐可视化**，最终产出可用 Chrome Trace / Perfetto 打开的 trace 文件。

关键约束：

- **时间对齐**：各机器墙钟（wall clock）无法精确同步，但产出的 span 必须能跨机器比较先后与重叠关系；
- **低开销**：记录的插桩点（start/stop）在训练热路径上，导出（事件同步 + 序列化 + 写盘）不能阻塞训练线程；
- **语义正确**：通信类操作实际运行在 NCCL 内部持有的 CUDA stream 上，事件必须记录到**同一条 stream** 才能量到真实耗时；
- **不破坏训练**：作为旁路观测系统，任何采集/导出异常都不得导致训练崩溃或 hang；
- **每机聚合**：同一台物理机上的多个 rank 的 span 应合并进同一个 trace 文件，减少文件数量并便于单机内对比。

系统对外提供两类 API：

- **高层 API**（`ndtimeline.api`）：`init_ndtimers`（初始化）、`flush`（导出并校准全局时钟）、`wait`（优雅退出）、`set_global_step` / `inc_step`（维护全局 step）；
- **低层 API**：`NDTimerManager` 及 `ndtimeit` / `ndtimeit_p2p` / `ndtimer` 装饰器等插桩原语，供框架内部（如 `vescale.pipe`）使用。

## 3. Implementation Challenges（实现挑战）

### 3.1 跨机器时间轴对齐（核心难点）

机器间的 Unix 时钟存在漂移，NTP 精度不足以判断毫秒级的通信重叠关系。ndtimeline 的解法是**模拟全局时钟**（`timer.py: GlobalReferenceTime`）：

- 初始化时做一次 `all_gather`，收集所有 rank 的当前 Unix 时间戳（ns），取**全局最小值** `initial_min_clock` 作为模拟时钟的纪元原点。由于 barrier + all_gather 本身是一个同步点，各 rank 时间戳的误差被限制在通信延迟内；
- 每个 rank 维护两个参考 CUDA event（round-robin 双缓冲）。`calibrate()` 时在默认 stream 上记录参考 event 并同步，记录此刻的 CPU 墙钟 `last_calibrated_at`；
- **CPU–GPU 时钟漂移修正**：相邻两次校准之间，比较两个参考 event 的 GPU `elapsed_time` 与 CPU `time.time()` 差值，得到残差系数 `gpu_clock_residual_coef = 1 - (gpu_time - cpu_time) / gpu_time`；
- 任意事件的绝对时间戳 = `gpu_time(ref → event) * gpu_clock_residual_coef + last_calibrated_at`。这样每个 GPU event 都被换算到"以全局最小时钟为原点、经漂移修正的墙钟"上，跨机器的 span 即可直接对齐。

挑战在于：校准本身需要同步（有开销），因此 `flush` 提供 `dynamic_calibrate`（至少每 30 分钟校准一次）与 `sequential_calibrate`（在主线程还是 flush 线程中校准）来控制精度与开销的平衡。

### 3.2 采集开销控制

- **CUDA Event 对象池**（`pool.py`）：`torch.cuda.Event` 的创建/销毁有成本且涉及 CUDA API。`CudaEventPool` 预分配 20 个 event（warmup 后入池），`get`/`release` 加锁复用；池耗尽时才新建；
- **异步 flush**（`timer.py: NDTimerManager.async_flush`）：flush 时把当前 timer 字典整体"换出"（用 meta 的拷贝重建一套新 timer 继续采集），旧 timer 交给 `ThreadPoolExecutor`（默认 3 worker）在后台做事件同步、耗时计算和 handler 派发。训练线程只做一次字典交换（期间临时 `gc.disable()` 避免 GC 触发 CUDA event 析构）；
- **带外进程聚合**（见 3.4）：写文件等重活完全移出训练进程；
- **级别过滤**（`NDMetricLevel`）：每个 timer 带级别（FRAMEWORK_INFO/USER_INFO/INFO/…/FRAMEWORK_TRACE/USER_TRACE/TRACE，有序枚举），低于 manager 级别的 metric 在 `start`/`stop` 入口直接返回，零开销；
- **未注册容错**：对未注册名字的 start/stop 只警告一次（记入 `_unregistered_timer_*` 列表后跳过），不反复打日志。

### 3.3 获取 NCCL 通信所用的 CUDA stream

要度量一次 P2P/集合通信的真实 GPU 耗时，CUDA event 必须记录到该通信实际运行的 stream 上。但 PyTorch 原生**不暴露** NCCL 使用的 stream——`ProcessGroupNCCL` 内部维护一个 `ncclStreams_` 池，按需懒分配。

ndtimeline 的解法是**给 PyTorch 打补丁**（`legacy/patches/patched_pytorch_v2.2.1_rc3.patch`），在 `ProcessGroupNCCL` 上新增两个 C++ 接口并绑定到 Python：

- `getP2PCudaStreamId(device_idx, peer, is_batched)`：batched 模式按 device key 查 `ncclStreams_`，非 batched 模式按 `getKeySendRecv(rank, peer)` 查；找到则返回 `stream.id()`，否则返回 -1；
- `getCollCudaStreamId(tensors)`：按 tensor 所在 device 的 key 查，返回 collective 使用的 stream id。

`stream.py` 中的 `get_nccl_p2p_stream` / `get_nccl_coll_stream` 把裸 stream id 包装成 `torch.cuda.Stream` 对象，并按 metric 名缓存到全局 `NCCL_STREAMS` 字典（每个 metric 固定对应一条 stream）。

**重要 caveat（代码与 README 均有强调）**：NCCL stream 是懒分配的，**第一次通信操作发生时 stream 才存在**，因此第一次通信必然查不到 stream（返回 -1，日志打印 `p2p stream is not available, skipped`），首个通信 span 会丢失，这是预期行为。这也意味着该补丁是 ndtimeline 通信度量的硬依赖——在未打补丁的 torch 上，`hasattr(nccl_backend, "get_p2p_cuda_stream_id")` 为假，通信类 metric 将全部退化为无 stream 记录。

### 3.4 单机聚合与训练进程减负

每个训练进程只承担"记录 + 序列化 + 发 socket"的轻活，聚合与落盘由独立的 **streamer 子进程**完成（`sock_streamer.py` + `handlers/sock_handler.py` + `binary_protocol.py`）：

- 每台机器上由 `local_rank == 0` 的进程通过 `torch.multiprocessing.spawn`（spawn 方式、daemon）拉起一个 streamer 进程；
- streamer 进程在 Unix domain socket（默认 `/opt/tiger/tmp/ndtimeline/ndtimeline.sock`）上跑 `ThreadingUnixStreamServer`，每连接一线程收包，投递进容量 500000 的 `queue.Queue`，再由消费线程调用 post handler（默认 `LocalTimelineNDHandler`）；
- 训练进程侧，`SockNDHandler` 作为 pre handler 在 flush 线程中把每条 metric 序列化后 `sendall` 到该 socket；连接失败带退避重试，server 退出后置 `server_exited` 不再发送——采集失败绝不影响训练；
- 传输用自定义**二进制封包协议**（`binary_protocol.py`）：

  ```
  +---------------------------------------------------------------+
  | Magic Number 1Byte (0x9C) | Protocol Version 1Byte | Reserved 2Byte |
  +---------------------------------------------------------------+
  |                   Payload Length 4Byte (little endian)        |
  +---------------------------------------------------------------+
  |          Payload (pickle protocol 4, 上限 128MiB)             |
  +---------------------------------------------------------------+
  | EOF Symbol 1Byte ('\n')                                       |
  +---------------------------------------------------------------+
  ```

  接收端 `recv_and_validate` 逐字段校验 magic/version/reserved/长度/EOF，支持粘包处理（`preload_data` 缓冲跨包残留字节）。

### 3.5 Metric 与训练 step 的关联

一个 metric 在一次 flush 周期内可能产生多条 span（例如一个 step 内多次 microbatch 的 send），需要把 span 归属到正确的 global step：

- **方式一（推荐）**：timer 注册时提供 `step_getter`，每条 span 记录时把当前 step 写进 tag（键 `_inner_global_step`）。`parse_record` 用 `itertools.groupby` 按 step 分组生成 record；
- **方式二**：不提供 `step_getter` 时，`flush` 传入 `step_range`，`parse_record` 要求 span 数量能被 step 数整除后**均分**到各 step（隐含"每 step 操作数相同"的假设，否则报 `NDHandlerError`，提示改用 `step_getter`）；
- 高层 API 还提供 `set_global_step` / `inc_step`，配合 manager 内部的 nature step getter 使用，免去用户写回调。

### 3.6 健壮性与资源管理

- 所有插桩入口（`start_timer`/`stop_timer`/`insert_record`）和 handler 调用都被 try/except 包裹，异常只记日志；
- `wait()` 先做 `dist.barrier()`，再等 flush futures（最多 10s），最后 sleep 3s 给 streamer 排空消息；
- 若干反直觉的资源处理均有注释说明：`_flush_timers` 末尾 `timer.meta = None` 防止 CudaTimer 因 meta 循环引用导致 event GC 失败；`LocalTimelineNDHandler` 切分缓冲时用 `record.copy()` 而非直接持有引用，同样是规避 GC 导致的内存泄漏；`loads`（pickle 反序列化）前后 `gc.disable/enable`；
- 双缓冲参考 event 使校准与正常记录互不踩踏。

## 4. Implementation Details（实现细节）

### 4.1 代码结构

```
legacy/vescale/ndtimeline/
├── __init__.py            # 包导出（try import _internal 兼容内部版）
├── api.py                 # 高层 API：init_ndtimers / flush / wait / set_global_step / inc_step
├── timer.py               # 核心：GlobalReferenceTime、NDMetricLevel、DeviceTimerMeta、
│                          #       DeviceTimer、NDTimerManager(Singleton)、ndtimeit 系列
├── pool.py                # CudaEventPool / DefaultEventPool（事件对象池）
├── stream.py              # get_nccl_p2p_stream / get_nccl_coll_stream（依赖 patched torch）
├── sock_streamer.py       # NDtimelineStreamer：streamer 子进程 + UDS server + 消费队列
├── binary_protocol.py     # 二进制封包/拆包协议
├── world_info.py          # WorldInfo = TopoInfo(rank/tp/dp/pp/ddp…) + TrainingInfo(role/trial/run)
├── predefined.py          # 预定义 metric 名（forward-compute、cross-mesh-send 等）
├── variables.py           # 常量：socket 路径/超时、event 池大小、tag 键、flush 特殊标记
├── logger.py              # NDTimelineLogger 单例（env VESCALE_NDTIMELINE_LOG_LEVEL 控制级别）
├── exceptions.py          # ProtocolValidationError / NDHandlerError
├── is_internal.py         # is_internal_vescale()：内部版/开源版能力门控
├── fsdp_patch.py          # patch_fsdp()（仅内部版有实现）
└── handlers/
    ├── handler_base.py        # NDHandler 抽象基类（dispatch_key、ignore_metrics、call_impl）
    ├── chrome_trace_event.py  # Chrome Trace Event Format 数据类（X/B/E/s-t-f/C/M 事件）
    ├── parser_handler.py      # parse_record + DeviceTimerStreamRecord（span → 结构化 record）
    ├── local_timeline_handler.py  # 单机聚合 → trace_step{min}_{max} JSON 文件
    ├── sock_handler.py        # SockNDHandler（训练进程侧 UDS 客户端，默认 pre handler）
    ├── logging_handler.py     # 日志摘要 handler
    ├── local_raw_handler.py   # 原始数据落盘（RotatingFileHandler，128MiB × 8）
    └── do_nothing_handler.py  # 空实现（占位/测试）
```

### 4.2 数据流（端到端）

```
训练热路径                    flush 线程（ThreadPoolExecutor）        streamer 子进程（local_rank 0 拉起）
─────────                    ──────────────────────────────         ────────────────────────────────────
ndtimeit_p2p(name, pg, peer)
 ├─ get_nccl_p2p_stream ────▶ 取 NCCL 内部 stream（patched torch）
 ├─ timer.start():
 │   event_pool.get() → event.record(stream)
 └─ timer.stop():
     event.record(stream)
          ……训练继续……
                             api.flush(step_range)
                              └─ async_flush:
                                  换出新 timer 字典（gc.disable 保护）
                                  submit _flush_timers ──────────────▶
                                    timer.elapsed():
                                      event.synchronize()
                                      duration = start.elapsed_time(stop)
                                      abs_ts = GlobalReferenceTime.elapsed_time(start)
                                      event 归池；tag 按 legal_tags 过滤
                                    对每个 handler 调用（dispatch_mode 过滤）:
                                      SockNDHandler:
                                        pickle + encode_package
                                        sendall → UDS ──────────────▶ MsgHandler 收包校验
                                                                        q.put(args)
                                                                      消费线程:
                                                                        LocalTimelineNDHandler
                                                                          parse_record → record 入 rank 缓冲
                                    发送 NDTIMELINE_FLUSH_SEPCIAL ───▶ 收到所有 local rank 的 special 后:
                                                                        dump_records() → Chrome Trace JSON
                                                                        （pid=dp_rank, tid=rank*10+类别序号,
                                                                          ts/dur 秒→微秒）
```

### 4.3 关键类与机制要点

- **`GlobalReferenceTime`**（类级状态的单例式工具类）：`init` 预热两个参考 event 并做首次 `calibrate`（含 all_gather 取 `initial_min_clock`）；`elapsed_time(end_event)` 是 span 绝对时间戳的唯一来源。注意它假设所有 rank 同时调用 `calibrate`（首次有 barrier），因此校准只能通过 `flush` 路径集体触发。
- **`DeviceTimer`**：一个 timer 可同时持有 GPU event pairs（`_event_pairs`）和 CPU 记录（`_extra_records`，`is_cpu_op=True` 时用 `time.time()`）。`elapsed()` 会把两类记录按起始时间**归并排序**成统一序列返回 `(total_elapsed, durations, since_starts, tags)`。带 stream 的 start 会先 `wait_stream(default_stream)`，保证事件捕获到从默认 stream  fork 出的通信。
- **`NDTimerManager.async_flush` 的换出机制**：`past_name2timer` 携带已有事件去后台处理；`fresh_name2timer` 由 meta 拷贝重建，训练线程立即切到新字典继续 start/stop——这是"记录与导出解耦"的关键。`next_iter_enabled` / `keep_timer_state` 决定换出后 timer 是否继续使能。
- **Handler 派发**：`NDHandler.dispatch_key` 默认为类名；timer 的 `dispatch_mode="selected"` + `dst_names` 可让某 metric 只发给指定 handler；`specified_extra[dispatch_key]` 提供按 handler 区分的附加字段。`ignore_metrics` 默认忽略 `special` 标记，`LocalTimelineNDHandler` 特意清空该忽略列表以接收 flush 分界信号。
- **`LocalTimelineNDHandler` 的分段落盘**：为每个 local rank 维护 `rank2steps`（deque of step-set）。每收到一个 rank 的 `special` 就追加一个新 set；当所有 rank 的 deque 长度 ≥ 2，说明上一段已收齐，把属于旧段的 record 落盘成 `trace_step{min}_{max}` 文件、缓冲中保留新段。落盘时按 metric 名查 `thread_sort_index` 表（main / pp → / pp ← / pp send-recv / tp collective / dp collective / other）决定 trace 中的线程分组，`tid = rank*10 + sort_index`（乘 10 给类别编号让位），`pid = dp_rank`。
- **WorldInfo**：`TopoInfo` 携带 rank、local_rank、tp/pp/dp/ddp 的 rank 与 size、world_size、ip，并做非负/正数校验；`TrainingInfo` 携带 role_id（RL 多角色）、trial_id、run_id。所有 record 都带这些维度，这就是"N-Dimension"的含义——trace 可按任意并行维度过滤/对比。
- **内外部版本门控**：`is_internal_vescale()`（能否 import `_internal`）决定 `fsdp` 模式、`MQNDHandler`（发到内部消息队列的 post handler）、`patch_fsdp` 等能力是否可用。开源版中 `init_ndtimers(mode="fsdp")` 直接抛 `NotImplementedError`；`fsdp` 模式的预置 timer（unshard-all-gather / grad-reduce-scatter / grad-all-reduce）在开源版也不会被注册。
- **与训练框架的集成点**：`vescale/pipe/p2p_communication.py` 用 `ndtimeit_p2p` 包裹 `_communicate`（RECV_FORWARD/RECV_BACKWARD/SEND_FORWARD/SEND_BACKWARD/SEND_FORWARD_RECV_BACKWARD/SEND_BACKWARD_RECV_FORWARD）；`vescale/pipe/_schedules/pipedream_flush.py` 与 `zero_bubble_v.py` 度量 CROSS_MESH_RECV/SEND，并用 `ndtimer` 装饰器度量 FORWARD_COMPUTE/BACKWARD_COMPUTE。

### 4.4 使用方式

```python
# 必须在 torch.cuda.set_device 和默认进程组初始化之后调用
from vescale.ndtimeline import init_ndtimers, flush, wait, inc_step

init_ndtimers(
    mode="hybrid",               # 开源版仅支持 hybrid（从 VESCALE_DEVICE_MESH 读 TP/DP/PP）
    step_getter=lambda: cur_step,  # 或用 inc_step()/set_global_step() 维护
    enable_streamer=True,        # fork streamer 子进程；默认 post handler 为 LocalTimelineNDHandler
    level=NDMetricLevel.DEBUG,   # 低于该级别的 metric 不记录
)

# ……训练循环……（框架插桩点自动记录；用户可用 ndtimeit/ndtimer 追加自定义 metric）
for step in range(n):
    train_one_step()
    inc_step()
    if step % 10 == 0:
        flush(step_range=range(step - 10, step))  # 异步导出 + 可选校准

wait()  # 训练结束时等待 flush 完成
```

产出的 `trace_step{min}_{max}` 文件是 Chrome Trace Event Format JSON，可直接拖入 https://ui.perfetto.dev/ 查看。若需跨机器汇总，可自行实现一个 post handler（如 MQHandler）把所有机器的 metric 送到中心存储后再生成 trace（内部版即通过 `MQNDHandler` 实现）。

## 5. How to Test This Feature（测试方法）

### 5.1 现有单元测试（无需 GPU）

单测位于 `legacy/test/ndtimeline/`：

| 测试文件 | 覆盖内容 |
|---|---|
| `test_metric_level.py` | `NDMetricLevel` 有序枚举的比较语义（INFO/DEBUG/TRACE 三档及 FRAMEWORK_/USER_ 前缀的大小关系） |
| `test_parser_handler.py` | `parse_record`：带 tag 的正常输入生成 record 且 step 正确；三个数组长度不一致时抛 `NDHandlerError` |
| `test_local_raw_handler.py` | `LocalRawNDHandler` 的 RotatingFileHandler 行为（按 chunk 滚动、backup 数量上限） |

运行方式（legacy 测试需从 `legacy/test/` 目录运行并把该目录加入 `PYTHONPATH`，因为测试间以顶层模块方式互相 import）：

```bash
cd legacy
pip3 install -r requirements.txt && pip3 install -e .

cd test
export PYTHONPATH=$(pwd):$PYTHONPATH

# 运行整个 ndtimeline 单测目录
pytest ndtimeline/ -s

# 运行单个测试文件 / 单个用例
pytest ndtimeline/test_parser_handler.py -s
pytest ndtimeline/test_parser_handler.py::test_normal_input_with_tags -s
```

注意：`test_local_raw_handler.py` 会在 `/opt/tiger/tmp/ndtimeline/`（`variables.py` 中的 `LOCAL_LOGGING_PATH`）下创建并清理滚动日志文件，需要该路径可写。

### 5.2 集成测试（需要多 GPU）

`legacy/test/parallel/pipeline/instruction/test_schedule.py` 中的流水线调度测试真实调用了 `init_ndtimers → flush → wait` 全流程（在 4 卡 PP 场景中随 `ScheduleEngine` 运行）。按 legacy 套件的方式运行：

```bash
cd legacy/test
export PYTHONPATH=$(pwd):$PYTHONPATH
export VESCALE_SINGLE_DEVICE_RAND=1
pytest parallel/pipeline/instruction/test_schedule.py -s -k <test_name>
```

（`legacy/scripts/run_test.sh` 会跑整个 legacy 测试集，并在每个文件前后 `pkill python3` 清理分布式测试的残留进程——本地跑多进程测试卡住时，残留进程通常是原因。）

### 5.3 手工端到端验证

最小验证路径（需要至少 1 台多卡机器、patched torch 才能看到通信 span）：

1. 在任意多进程训练脚本中（`torch.cuda.set_device` 与 `dist.init_process_group` 之后）调用 `init_ndtimers(enable_streamer=True)`，训练若干 step，周期性 `flush(step_range=...)`，结束时 `wait()`；
2. 确认日志出现 `ndtimeline streamer started` 与各 rank 的 `calibrate ...` 信息；预期看到若干 `p2p stream is not available, skipped`（首个通信 span 丢失，属预期）；
3. 在 cwd 下得到 `trace_step{min}_{max}` 文件，拖入 Perfetto，检查：每个 dp_rank 一个进程行、每个 rank 按 main/pp/tp/dp 分线程行、forward-compute 与 send/recv span 的时间先后关系符合调度预期；
4. 验证低开销路径：设置 `level=NDMetricLevel.INFO`（高于预置 timer 的默认 FRAMEWORK_DEBUG），确认对应 span 不再出现——级别过滤生效。

### 5.4 针对子系统的新测试建议

- **协议层**：可构造 payload 直接测试 `encode_package`/`recv_and_validate` 的粘包与坏 magic/坏 EOF/超长 payload 分支（纯 CPU，适合单测）；
- **时钟换算**：`GlobalReferenceTime` 依赖真实 CUDA，可在单卡上验证 `elapsed_time` 单调性与 `gpu_clock_residual_coef` 的合理范围；
- **handler 扩展**：新写一个 `NDHandler` 子类时，参照 `ParserNDHandler` 的单测方式直接构造 `call_impl` 入参验证，无需启动分布式环境。
