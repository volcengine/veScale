# ndtimeline 实现细节(API、接入、Tutorial 与实现原理)

> 本文档是 [`ndtimeline-design.md`](./ndtimeline-design.md) 的深入补充,聚焦**怎么用**和**怎么实现的**。设计文档侧重动机、问题定义与挑战;本文档侧重 API 参考、接入步骤、可运行的 PyTorch tutorial,以及逐组件的实现原理剖析。
>
> 代码位置:`legacy/vescale/ndtimeline/`(属于旧版 veScale,当前仓库活跃代码为新版 DTensor)。

---

## 目录

1. [API 定义](#1-api-定义)
2. [如何接入 ndtimeline](#2-如何接入-ndtimeline)
3. [PyTorch Tutorial(可运行)](#3-pytorch-tutorial可运行)
4. [实现原理详解](#4-实现原理详解)
5. [局限与注意事项](#5-局限与注意事项)

---

## 1. API 定义

ndtimeline 的 API 分两层:**高层 API**(`ndtimeline/api.py`,面向最终用户,封装好默认行为)和**低层 API**(`timer.py` 等,面向框架开发者,完全可控)。高层 API 的 `init_ndtimers` 依赖 veScale 的全局 device mesh(`VESCALE_DEVICE_MESH`),因此**纯 PyTorch 场景应直接使用低层 API**(见 §3)。

### 1.1 高层 API(`ndtimeline.api`)

#### `init_ndtimers(...)`

一次性完成:拉起 streamer 子进程、注入默认 handler、构造 `NDTimerManagerSingleton`、注册预置 timer。

```python
def init_ndtimers(
    rank: Optional[int] = None,                 # 全局 rank,None 时用 dist.get_rank()
    mode: Literal["fsdp", "hybrid"] = "hybrid", # 并行模式;fsdp 仅内部版支持
    wrapped_fsdp_module=None,                   # fsdp 模式用(内部版)
    device_mesh=None,                           # fsdp 模式用(内部版)
    mesh_shape: Optional[Tuple[int, ...]] = None,  # fsdp 模式用(内部版)
    local_rank: Optional[int] = None,           # None 时用 VESCALE_DEVICE_MESH.get_local_rank()
    step_getter: Optional[Callable[[], int]] = None,  # 返回当前 global step 的回调;None 时用 set_global_step/inc_step
    enable_streamer: bool = True,               # 是否 fork streamer 子进程以启用 post handler
    n_rank_per_host: Optional[int] = None,      # 单机 GPU 数,None 时用 torch.cuda.device_count()
    pre_handlers: Optional[List[NDHandler]] = None,   # 训练进程内、flush 线程触发的 handler
    post_handlers: Optional[List[NDHandler]] = None,  # streamer 进程内触发的 handler
    user_spcified_timers: Optional[List[DeviceTimerMeta]] = None,  # 用户自定义 timer
    level: NDMetricLevel = NDMetricLevel.DEBUG, # 级别阈值,低于它的 metric 不记录
    ip: str = "0.0.0.0",                        # 本机 IP
    **kwargs,
)
```

**调用前提(硬性要求)**:必须在 `torch.cuda.set_device(...)` 和默认进程组 `dist.init_process_group(...)` **之后**调用。

**默认行为**:
- `mode="hybrid"`:从 `VESCALE_DEVICE_MESH` 读取 TP/DP/PP 的 size 与 rank,并断言 `tp*dp*pp == world_size`。
- `mode="fsdp"`:仅内部版;会 `patch_fsdp()`,并从 process group / device_mesh / mesh_shape 推算 dp/ddp 拓扑。
- `enable_streamer=True` 时:
  - `local_rank == 0` 的进程拉起 streamer 子进程;若未给 `post_handlers`,默认注入 `LocalTimelineNDHandler(n_rank_per_host)`(在本机生成 Perfetto trace 文件)。
  - 若 `pre_handlers` 里没有 `SockNDHandler`,自动追加一个(负责把 metric 通过 Unix socket 发给 streamer)。
- `hybrid` 模式注册的预置 timer:`forward-compute`、`backward-compute`、`forward/backward-recv`、`forward/backward-send`、`forward-send-backward-recv`、`backward-send-forward-recv`、`cross-mesh-recv/send`(均为 GPU op,`is_cpu_op=False`)。
- `fsdp` 模式注册的预置 timer:`unshard-all-gather`、`grad-reduce-scatter`、`grad-all-reduce`、`forward-compute`、`backward-compute`。

#### `flush(...)`

把已采集的 metric 异步导出到 handler,并可选地校准全局时钟。

```python
def flush(
    step_range: Optional[range] = None,  # 这批 metric 归属的 step 范围;给了 step_getter 时可忽略
    next_iter_enabled: bool = True,      # flush 后 timer 是否继续使能
    submit2handler: bool = True,         # False 表示丢弃 metric(只清空不导出)
    dynamic_calibrate: bool = False,     # 是否"至少每 30 分钟校准一次时钟"
    keep_timer_state: bool = False,      # True 时保持各 timer 现有 enable/disable 状态(忽略 next_iter_enabled)
    sequential_calibrate: bool = True,   # True=在主线程校准;False=放到 flush 后台线程校准
)
```

#### `wait()`

优雅退出:先 `dist.barrier()`,再等所有后台 flush future(最多 10s),最后 `sleep(3)` 给 streamer 排空消息。**应在训练结束时(`main.py` 末尾)调用**。

#### `set_global_step(step)` / `inc_step(step=1)`

当 `step_getter=None` 时,用这两个 API 手动维护 manager 内部的 global step。每条 span 记录时会把当前 step 写进 tag。

### 1.2 低层 API(`timer.py`)

#### `NDTimerManager` / `NDTimerManagerSingleton`

`NDTimerManagerSingleton` 是 `NDTimerManager` 的单例子类,**所有 context manager(`ndtimeit` 等)都作用于这个单例**。

```python
NDTimerManagerSingleton(
    world_info: WorldInfo,
    handlers: Optional[List[Callable]] = None,   # pre handler 列表
    max_workers: int = 3,                        # 后台 flush 线程池大小
    device_id: Optional[int] = None,             # None 时用当前 CUDA device
    init_cuda_dist: bool = True,                 # True 时初始化 event 池 + 全局时钟(含集合通信)
    metric_level: NDMetricLevel = NDMetricLevel.TRACE,  # 级别阈值
    is_nature_step: bool = True,                 # True 时,无 step_getter 的 timer 自动用 manager 内部 step
)
```

核心方法:

| 方法 | 说明 |
|---|---|
| `register_timers(metas: List[DeviceTimerMeta])` | 从 meta 列表创建 `DeviceTimer`。**只能调用一次**(内部断言) |
| `start_timer(name, tag=None)` | 开始计时;`tag` 里可带 `NDTIMELINE_STREAM_KEY` 指定 stream |
| `stop_timer(name, tag=None)` | 停止计时 |
| `insert_record(name, start_ts, duration, tag=None)` | 直接插入一条已完成记录(不经 event),用于补录 |
| `async_flush(step_range, ...)` | 异步导出(见 §4.5) |
| `wait()` | 等待后台 flush 完成 |
| `clear()` | 丢弃当前所有未导出记录 |
| `disable_and_save()` / `recover_from_history()` | 临时禁用全部 timer 并恢复(期间关 GC) |
| `global_step`(property) | 读/写当前 step |
| `is_initialized()`(classmethod) | 单例是否已构造 |

#### `DeviceTimerMeta`(timer 的声明式配置)

```python
@dataclass
class DeviceTimerMeta:
    name: str = ""
    is_cpu_op: bool = False                 # True=用 time.time() 计 CPU 耗时;False=用 CUDA event
    legal_tags: List[str] = []              # 允许保留进记录的 tag 键白名单
    step_getter: Optional[Callable] = None  # 返回 step 的回调
    enabled: bool = True
    level: NDMetricLevel = NDMetricLevel.FRAMEWORK_DEBUG
    device_id: int = -1                     # -1 时:GPU op 用当前 device,CPU op 用 0
    dispatch_mode: Literal["selected", "all"] = "all"  # selected=只发给 dst_names 指定的 handler
    dst_names: List[str] = []               # dispatch_mode="selected" 时的目标 handler dispatch_key
    specified_extra: Dict[str, Any] = {}    # 按 handler dispatch_key 区分的附加字段
    common_extra: Dict[str, Any] = {}       # 对所有 handler 的附加字段
```

#### `NDMetricLevel`(级别过滤)

有序枚举,值越大越"啰嗦"。`timer.level > manager.metric_level` 时该 metric 在 `start/stop` 入口直接返回(零开销)。

```
FRAMEWORK_INFO(2) < USER_INFO(3) < INFO(4)
< FRAMEWORK_DEBUG(12) < USER_DEBUG(13) < DEBUG(14)
< FRAMEWORK_TRACE(102) < USER_TRACE(103) < TRACE(104)
```

预置 timer 默认 `FRAMEWORK_DEBUG`;manager 默认阈值 `DEBUG`(低层)/`DEBUG`(高层),所以默认能收到预置 metric。

#### 插桩原语(context manager / 装饰器)

```python
ndtimeit(name: str, tag: Optional[Dict] = None)
    # 通用计时:在默认/当前 stream 上记录 start/stop event。可重入。

ndtimeit_p2p(name, nccl_pg, peer, is_batched=True, tag=None)
    # P2P 通信计时:先通过 patched torch 取 NCCL p2p stream,把 stream 塞进 tag,
    # 再 start/stop。事件记录到真实通信 stream 上,才能量到真实耗时。

ndtimeit_coll(name, pg, tensor, tag=None)
    # 集合通信计时:取 NCCL coll stream(依赖 patched torch),同上。
    # 注意:此函数未在包级 __init__ 导出,需 from vescale.ndtimeline.timer import ndtimeit_coll

ndtimer(metric: str, tags: Optional[Dict] = None)
    # 函数装饰器,等价于用 ndtimeit 包住整个函数体。
```

未初始化单例时,上述 context manager 都是**空操作**(直接 `yield`),因此插桩代码可以在不开启 ndtimeline 时安全保留。

### 1.3 Handler API(`handlers/`)

#### `NDHandler`(抽象基类)

```python
class NDHandler(ABC):
    def __init__(self, designated_key="", ignore_metrics=None):
        # dispatch_key 默认为类名;ignore_metrics 默认忽略特殊 flush 标记 NDTIMELINE_FLUSH_SEPCIAL

    def __call__(self, metric_name, elapsed, recent_elapsed_raw_parts,
                 recent_since_start_raw_parts, tags, step_range, world_info, extra):
        # 过滤 ignore_metrics 后调用 call_impl

    @abstractmethod
    def call_impl(self, metric_name, elapsed, recent_elapsed_raw_parts,
                  recent_since_start_raw_parts, tags, step_range, world_info, extra): ...
```

各参数含义:
- `metric_name`:metric 名(或特殊标记 `"special"`)。
- `elapsed`:本周期该 metric 的总耗时(秒)。
- `recent_elapsed_raw_parts`:每条 span 的耗时列表(秒)。
- `recent_since_start_raw_parts`:每条 span 的**绝对起始时间戳**列表(秒,基于模拟全局时钟)。
- `tags`:每条 span 的 tag(已经过 `legal_tags` 过滤)。
- `step_range`:本批归属的 step 范围。
- `world_info`:`WorldInfo`(拓扑 + 训练元信息)。
- `extra`:timer 的附加字段。

#### 内置 handler

| Handler | 运行位置 | 作用 |
|---|---|---|
| `SockNDHandler` | 训练进程(pre) | 把每条 metric 用二进制协议经 Unix socket 发给 streamer。**默认 pre handler** |
| `LocalTimelineNDHandler` | streamer 进程(post) | 聚合本机所有 rank,生成 Perfetto trace 文件 `trace_step{min}_{max}`。**默认 post handler** |
| `ParserNDHandler` | 任意 | 把原始入参解析成结构化 `DeviceTimerStreamRecord` 列表 |
| `LoggingNDHandler` | 任意 | 每个 metric 打一行日志(总耗时),最简单的验证手段 |
| `LocalRawNDHandler` | 任意 | 原始数据写滚动日志文件(128MiB × 8) |
| `DoNothingNDHandler` | 任意 | 空实现(占位/测试) |

自定义 handler:继承 `NDHandler`,实现 `call_impl` 即可。若只想接收特定 metric,配合 timer 的 `dispatch_mode="selected"` + `dst_names=[你的 dispatch_key]`。

### 1.4 数据结构

#### `WorldInfo`(`world_info.py`)

```python
WorldInfo(
    rank, local_rank,                 # 必填
    dp_rank=0, ddp_rank=0, tp_rank=0, pp_rank=0,
    dp_size=1, ddp_size=1, tp_size=1, pp_size=1, world_size=1,
    ip="0.0.0.0",
    role_id=0, run_id=0, trial_id=0,  # 训练元信息(RL 多角色 / 试验 / 运行)
    **extra_meta,
)
```
内部拆成 `TopoInfo`(拓扑,带非负/正数校验)+ `TrainingInfo`(训练元信息)+ `extra_info`,支持 `world_info["rank"]` 式访问。

#### `DeviceTimerStreamRecord`(`handlers/parser_handler.py`)

`parse_record` 的输出,一条结构化记录:`ts/rank/metric/step/avg_dur/start_ts[]/duration[]/tp_rank/pp_rank/dp_rank/ip/role_id/trial_id/run_id`。`to_dict()` 输出供下游(如内部 MQ)消费的字段。

### 1.5 关键常量(`variables.py`)

| 常量 | 值 | 含义 |
|---|---|---|
| `SOCK_PARENT_DIR` | `/opt/tiger/tmp/ndtimeline` | streamer socket 与日志目录(**需可写**) |
| `SOCK_PATH` | `.../ndtimeline.sock` | Unix domain socket 路径 |
| `SOCK_TIMEOUT_CLIENT` | `2.0`s | socket 客户端超时 |
| `DEFAULT_CUDA_EVENT_POOL_SIZE` | `20` | event 池初始大小 |
| `NDTIMELINE_INNER_GLOBAL_STEP_KEY` | `_inner_global_step` | span 里 step 的 tag 键 |
| `NDTIMELINE_STREAM_KEY` | `stream_key` | tag 里指定 stream 的键 |
| `NDTIMELINE_FLUSH_SEPCIAL` | `special` | flush 分界特殊标记 |

---

## 2. 如何接入 ndtimeline

### 2.1 前提条件

1. **安装 legacy veScale**:`cd legacy && pip install -e .`。
   注意:即使只用 ndtimeline,`import vescale.ndtimeline` 也会经由 `api.py` 传递性导入 `vescale.dtensor` 和 `vescale.devicemesh_api`,所以必须完整安装 legacy 包。
2. **CUDA 环境**:计时核心基于 `torch.cuda.Event`,需要可用 GPU。
3. **已初始化分布式**:默认进程组已建立(`dist.init_process_group`),且已 `torch.cuda.set_device`。全局时钟校准需要集合通信(barrier + all_gather)。
4. **(可选,精确通信计时)patched PyTorch**:`ndtimeit_p2p`/`ndtimeit_coll` 依赖给 PyTorch 打的补丁(`legacy/patches/patched_pytorch_v2.2.1_rc3.patch`,新增 `ProcessGroupNCCL.get_p2p_cuda_stream_id`/`get_coll_cuda_stream_id`)。**未打补丁时**,这些函数取不到 NCCL stream,会退化为在默认 stream 上记录(量到的是"发起时刻"而非真实通信耗时),并打印 `p2p stream is not available, skipped`。**计算类计时不受影响**。
5. **可写的 socket 目录**:默认 `/opt/tiger/tmp/ndtimeline` 需存在且可写(streamer 与 SockNDHandler 都用它)。

### 2.2 两条接入路径

**路径 A:legacy veScale 训练(hybrid/fsdp 模式)——用高层 API**

veScale 的 pipeline / FSDP 框架内部已经在关键路径插好了桩(`pipe/p2p_communication.py` 用 `ndtimeit_p2p`,`pipe/_schedules/*.py` 用 `ndtimer`/`ndtimeit_p2p`)。用户只需:

```python
# 1) 先初始化 veScale 全局 device mesh(含 DP/TP/PP 维度)
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
VESCALE_DEVICE_MESH.init_device_mesh(
    "cuda", mesh_shape=(dp, pp, tp), mesh_dim_names=("DP", "PP", "TP"))

# 2) 初始化 ndtimeline(会自动读 device mesh、拉起 streamer、注册预置 timer)
from vescale.ndtimeline import init_ndtimers, flush, wait, inc_step
init_ndtimers(mode="hybrid", step_getter=lambda: global_step, enable_streamer=True)

# 3) 训练循环中维护 step、周期性 flush
for step in range(n):
    train_one_step()
    inc_step()                 # 或依赖 step_getter
    if step % FLUSH_INTERVAL == 0:
        flush(step_range=range(step - FLUSH_INTERVAL + 1, step + 1))

# 4) 结束时优雅退出
wait()
```

**路径 B:自定义 / 纯 PyTorch 框架——用低层 API**

不依赖 veScale device mesh,自己构造 `WorldInfo` 和 `NDTimerManagerSingleton`,自己插桩。完整可运行示例见 §3。

### 2.3 插桩点选择

| 想度量 | 用什么 | 说明 |
|---|---|---|
| 一段 GPU 计算(forward/backward) | `with ndtimeit("forward-compute"):` | 在默认 stream 记录 |
| 一次 P2P send/recv | `with ndtimeit_p2p(name, pg, peer, is_batched):` | 需 patched torch 才精确 |
| 一次集合通信 | `ndtimeit_coll(name, pg, tensor)` | 需 patched torch 才精确 |
| 整个函数 | `@ndtimer("metric-name")` | 装饰器 |
| CPU 操作(数据加载等) | timer meta 设 `is_cpu_op=True`,再 `ndtimeit` | 用 `time.time()` 计时 |
| 已知起止时间的补录 | `manager.insert_record(name, start_ts, duration, tag)` | 不经 event |

### 2.4 自定义 timer 与 handler

```python
from vescale.ndtimeline import DeviceTimerMeta, NDMetricLevel
from vescale.ndtimeline.handlers import NDHandler

# 自定义 timer:USER_DEBUG 级别,只发给指定 handler
metas = [DeviceTimerMeta(
    name="my-metric", is_cpu_op=False,
    level=NDMetricLevel.USER_DEBUG,
    dispatch_mode="selected", dst_names=["MyHandler"],
    common_extra={"source": "my_module"},
)]
NDTimerManagerSingleton().register_timers(metas)   # 注意:只能 register 一次

# 自定义 handler
class MyHandler(NDHandler):
    def call_impl(self, metric_name, elapsed, parts, since_parts, tags, step_range, world_info, extra):
        ...  # 任意处理:写库、发 MQ、统计等
```

### 2.5 关机顺序

`flush(...)`(把最后一批导出)→ `wait()`(barrier + 等后台 future + 给 streamer 3s 排空)。漏掉 `wait()` 可能丢失最后一批数据。

---

## 3. PyTorch Tutorial(可运行)

下面是一个**自包含的 DDP 示例**,不依赖 veScale device mesh,直接用低层 API。它在 2 个进程(2 卡)上跑一个简单 MLP 的 DDP 训练,度量 forward / backward / 一次显式 allreduce,并在本机生成 Perfetto trace 文件。

### 3.1 完整代码 `ndtimeline_tutorial.py`

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ---- ndtimeline 低层 API ----
from vescale.ndtimeline import (
    NDTimerManagerSingleton, DeviceTimerMeta, NDMetricLevel,
    ndtimeit, flush, wait, inc_step,
)
from vescale.ndtimeline.world_info import WorldInfo
from vescale.ndtimeline.handlers import SockNDHandler, LocalTimelineNDHandler
from vescale.ndtimeline.sock_streamer import NDtimelineStreamer

# 预定义的 metric 名(也可直接用字符串)
FORWARD_COMPUTE = "forward-compute"
BACKWARD_COMPUTE = "backward-compute"
GRAD_ALLREDUCE = "grad-allreduce"
DATA_LOAD = "data-load"


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def init_ndtimeline(rank, local_rank):
    world_size = dist.get_world_size()
    n_rank_per_host = torch.cuda.device_count()

    # 1) 仅 local_rank 0 拉起 streamer 子进程,默认 post handler 生成本机 trace 文件
    NDtimelineStreamer.init(
        local_rank,
        [LocalTimelineNDHandler(n_rank_per_host=n_rank_per_host)],
    )

    # 2) 构造单例 manager。init_cuda_dist=True 会初始化 event 池 + 全局时钟
    #    (内部含 barrier/all_gather,必须所有 rank 集体执行到这里)
    world_info = WorldInfo(
        rank=rank, local_rank=local_rank,
        dp_rank=rank, dp_size=world_size, world_size=world_size,
        ip="0.0.0.0",
    )
    NDTimerManagerSingleton(
        world_info,
        handlers=[SockNDHandler()],        # pre handler:经 socket 发给 streamer
        init_cuda_dist=True,
        metric_level=NDMetricLevel.DEBUG,  # 阈值:收到 FRAMEWORK_DEBUG 及以上的 metric
    )

    # 3) 注册 timer(只能注册一次)
    NDTimerManagerSingleton().register_timers([
        DeviceTimerMeta(FORWARD_COMPUTE, is_cpu_op=False, level=NDMetricLevel.FRAMEWORK_DEBUG),
        DeviceTimerMeta(BACKWARD_COMPUTE, is_cpu_op=False, level=NDMetricLevel.FRAMEWORK_DEBUG),
        DeviceTimerMeta(GRAD_ALLREDUCE, is_cpu_op=False, level=NDMetricLevel.FRAMEWORK_DEBUG),
        DeviceTimerMeta(DATA_LOAD, is_cpu_op=True, level=NDMetricLevel.FRAMEWORK_DEBUG),  # CPU op 示例
    ])


class MLP(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    def forward(self, x):
        return self.net(x)


def main():
    rank, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    init_ndtimeline(rank, local_rank)

    model = DDP(MLP().to(device), device_ids=[local_rank])
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    dim = 64
    num_steps = 10
    flush_every = 5

    for step in range(num_steps):
        inc_step()  # 维护 manager 内部 global step(每条 span 会带上它)

        # ---- 度量一段 CPU 操作(数据加载)----
        with ndtimeit(DATA_LOAD):
            x = torch.randn(32, dim, device=device)

        optim.zero_grad()

        # ---- 度量 GPU 计算 ----
        with ndtimeit(FORWARD_COMPUTE):
            y = model(x)
            loss = y.sum()
        with ndtimeit(BACKWARD_COMPUTE):
            loss.backward()

        # ---- 度量一次显式集合通信 ----
        # 注:精确的 allreduce 计时需 ndtimeit_coll + patched torch;
        #     这里用通用 ndtimeit(在默认 stream 记录)演示流程。
        grad = model.module.net[0].weight.grad
        with ndtimeit(GRAD_ALLREDUCE):
            dist.all_reduce(grad, op=dist.ReduceOp.AVG)

        optim.step()

        # ---- 周期性 flush:把 [step-flush_every+1, step] 这批导出 ----
        if (step + 1) % flush_every == 0:
            flush(step_range=range(step - flush_every + 1, step + 1))
            if rank == 0:
                print(f"[step {step}] flushed")

    # ---- 优雅退出 ----
    wait()
    if rank == 0:
        print("done; look for trace_step*_*.json in the launch directory")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### 3.2 运行

```bash
# 0) 安装 legacy veScale(只需一次)
cd legacy && pip install -e .

# 1) 确保 socket 目录可写(默认 /opt/tiger/tmp/ndtimeline)
sudo mkdir -p /opt/tiger/tmp/ndtimeline && sudo chmod 777 /opt/tiger/tmp/ndtimeline
#   (或改用一个你有权限的目录:见 §5 关于 SOCK_PARENT_DIR 的说明)

# 2) 用 torchrun 起 2 个进程(2 卡)
torchrun --nproc_per_node=2 ndtimeline_tutorial.py
```

运行时会看到日志 `ndtimeline streamer started`、各 rank 的 `calibrate ...` 信息,以及若干 `p2p stream is not available, skipped`(因为没有 patched torch,且本例用的是通用 `ndtimeit`,属正常)。结束后在**启动目录**生成 `trace_step0_9` 之类的文件。

### 3.3 查看 trace

把生成的 `trace_step{min}_{max}` 文件拖入 <https://ui.perfetto.dev/>。你会看到:

- 每个 dp_rank 一个 **process**(本例 dp_size=2,故 2 个 process);
- 每个 rank 按类别分成多个 **thread**:`main`(forward/backward-compute)、`dp collective`(grad-allreduce 等)、`other` 等;
- 每个 span 的宽度是耗时,横轴是**跨进程对齐的统一时间轴**(模拟全局时钟),因此可以直观比较两个 rank 的 forward 是否一前一后、allreduce 是否在等更慢的那个 rank。

### 3.4 最小冒烟测试(不想开 streamer / socket 时)

如果只想确认"插桩 + 计时"跑通、暂不需要 trace 文件,可以把 handler 换成 `LoggingNDHandler` 并跳过 streamer:

```python
from vescale.ndtimeline.handlers import LoggingNDHandler
NDTimerManagerSingleton(world_info, handlers=[LoggingNDHandler()],
                        init_cuda_dist=True, metric_level=NDMetricLevel.DEBUG)
NDTimerManagerSingleton().register_timers([...])
# 训练 + flush;每个 metric 每次 flush 会打一行总耗时日志
```

这条路不需要 `/opt/tiger/tmp` 可写,适合快速验证。

---

## 4. 实现原理详解

### 4.1 总体架构与数据流

```
训练热路径                      flush 线程(ThreadPoolExecutor)          streamer 子进程(local_rank 0 拉起)
─────────                      ──────────────────────────────           ────────────────────────────────────
ndtimeit(name):
  pool.get() → event.record()      ……训练继续……
  ... 被测代码 ...
  event.record()
                               flush(step_range)
                                └─ async_flush:
                                    换出新 timer 字典(gc.disable 保护)
                                    submit _flush_timers ───────────────▶
                                      timer.elapsed():
                                        event.synchronize()
                                        dur = start.elapsed_time(stop)         # ms→s
                                        abs = GlobalReferenceTime.elapsed_time(start)  # 模拟全局时钟
                                        event 归池;tag 按 legal_tags 过滤
                                      逐 handler 调用(dispatch 过滤):
                                        SockNDHandler:
                                          pickle + encode_package
                                          sendall → Unix socket ─────────────▶ MsgHandler 收包+校验
                                                                                 q.put(args)
                                                                               消费线程 internal_queue_consume:
                                                                                 LocalTimelineNDHandler
                                                                                   parse_record → record 入缓冲
                                      发送特殊标记 "special" ────────────────▶ 收齐本机所有 rank 的 special:
                                                                                 dump_records() → Chrome Trace JSON
                                                                                 (pid=dp_rank, tid=rank*10+类别,
                                                                                  ts/dur 秒→微秒)
```

设计主线:**记录与导出解耦、导出与落盘解耦**。热路径只记 event;导出(同步+计算+序列化)在后台线程;落盘/聚合在独立子进程。三层各自异步,把对训练关键路径的干扰降到最低。

### 4.2 GlobalReferenceTime:模拟全局时钟(核心)

这是整个系统最精巧的部分,解决"跨机器时间轴对齐"。位于 `timer.py`。

**问题**:各机器墙钟有漂移,NTP 精度不足以判断毫秒级通信重叠。需要一个"全局一致的时间坐标系"。

**三个关键量**:
- `initial_min_clock`(ns):全局时钟的**纪元原点**。
- `last_calibrated_at`(ms):本 rank 最近一次校准时刻的墙钟。
- `gpu_clock_residual_coef`:GPU 时钟相对 CPU 墙钟的漂移修正系数(≈1)。

**初始化 `init(world_sz, device)`**:
1. 在当前 device 上创建 **2 个参考 CUDA event**(双缓冲),先 record + synchronize 做 warmup。
2. 调 `calibrate()` 完成首次校准。
3. 置 `initialized=True`。

**校准 `calibrate()`**(round-robin 切换两个参考 event):
1. `ref_pointer` 指向下一个参考 event;`torch.cuda.synchronize()` 后在该 event 上 `record` 到默认 stream 并 `synchronize`,随即读 CPU 墙钟 `ts_ns = time.time_ns()`。这样这个 GPU event 就和一个 CPU 墙钟时刻绑定。
2. **首次校准**(`not initialized`):先 `dist.barrier()` 同步所有 rank,然后 `all_gather` 收集各 rank 的 `ts_ns`,取**全局最小值**作为 `initial_min_clock`。
   > 为什么取 min 可行?barrier+all_gather 本身是一个同步点,各 rank 读到 `ts_ns` 的误差被限制在集合通信延迟内;取 min 作为共同原点,之后每个 rank 的绝对时间都是"相对这个原点的偏移",天然可比。
3. `clock_diff = (ts_ns - initial_min_clock) / 1e6`(本 rank 相对原点的偏移,ms)。
4. **CPU–GPU 漂移修正**:计算距上次校准的 CPU 墙钟间隔 `cpu_time`;若 `initialized` 且 `2e3 < cpu_time < 200000e3`(2 秒~200000 秒之间,避免间隔太短或太久导致测不准),则用两个参考 event 的 GPU `elapsed_time` 得到 `gpu_time`,算出
   `gpu_clock_residual_coef = 1 - (gpu_time - cpu_time) / gpu_time`。
   > 含义:GPU 时钟(`elapsed_time` 所基于)与 CPU 墙钟会有微小速率差,用这个系数把 GPU 时间换算成墙钟时间。
5. 更新 `last_calibrated_at = ts`。

**换算 `elapsed_time(end_event)`**:
```python
gpu_time = ref_events[ref_pointer].elapsed_time(end_event)  # 参考点→目标事件的 GPU 时长(ms)
return gpu_time * gpu_clock_residual_coef + last_calibrated_at  # 修正后 + 参考点墙钟 = 目标墙钟(ms)
```
即:任意 GPU event 的**绝对墙钟时间** = (它距最近参考点的 GPU 时长 × 漂移系数) + 参考点的墙钟时刻。这样所有 rank 的 event 都被换算到"以 `initial_min_clock` 为原点、经漂移修正"的统一墙钟上。

**何时校准**:`flush` 的 `dynamic_calibrate=True` 时,若距上次校准超过 30 分钟就触发;`sequential_calibrate=True` 在主线程校准,否则放到 flush 后台线程。首次校准在 `init` 里必然发生(含 barrier)。

### 4.3 CudaEventPool:event 对象池(`pool.py`)

`torch.cuda.Event` 的创建/销毁有成本。`CudaEventPool` 预分配 `DEFAULT_CUDA_EVENT_POOL_SIZE=20` 个 event(`enable_timing=True, blocking=True`),每个先 `record()` 一次做 warmup 后入 `deque`。

- `get(tag)`:加锁从池里 `popleft`;池空才新建。把 tag 拷进 `event.tag`。
- `release(event)`:加锁归还。

`DefaultEventPool` 是其类级单例包装。`DeviceTimer` 的每次 start/stop 都从池里取 event,用完归还,避免频繁创建。

### 4.4 DeviceTimer:单个 metric 的计时器(`timer.py`)

状态:`_event_pairs`(GPU op 的 `[start_event, stop_event]` 列表)、`_extra_records`(CPU op 或补录的 `[start_ts, duration, tag]` 列表)、`_stream`、`_started`。

**`start(stream, tag, level)`**:
- 级别不够(`meta.level > level`)或未使能 → 直接返回。
- 若 `is_cpu_op`:记录 `[time.time(), None, tag]` 到 `_extra_records`。
- 否则从池取 start_event;若传了 `stream`,先 `stream.wait_stream(default_stream)`(保证 event 捕获到从默认 stream fork 出的通信),再 `event.record(stream=...)`。append `[start_event, None]`。

**`stop(tag, level)`**:CPU op 补上 duration;GPU op 取 stop_event record 并填进最近那对的末位。

**`elapsed(reset=True)`**(在 flush 线程调用,这是唯一会同步 GPU 的地方):
1. 对每对 event:`synchronize` 后,
   - `single_elapsed = start.elapsed_time(stop) / 1e3`(耗时,秒);
   - `single_since = GlobalReferenceTime.elapsed_time(start) / 1e3`(绝对起始,秒,见 §4.2);
   - 合并 start/stop 的 tag,并**只保留 `legal_tags` 白名单内的键**;
   - 两个 event 归还池。
2. 若有 `_extra_records`(CPU/补录),按起始时间**归并插入**到上面两个序列里(GPU 与 CPU 记录合并成统一时间序列)。
3. 返回 `(total_elapsed, durations[], since_starts[], tags[])`,`reset` 清空状态。

> `insert_record(start_ts, duration, tag)`:不走 event,直接塞一条已完成记录(用于把外部已知时间段补录进时间轴)。

### 4.5 NDTimerManager:注册、计时与异步 flush(`timer.py`)

**注册 `register_timers(metas)`**:
- 每个 meta:`device_id==-1` 时,GPU op 用当前 device、CPU op 用 0;若 `step_getter is None and is_nature_step`,自动注入一个返回 `self._cur_global_step` 的 getter。
- 由 meta 构造 `DeviceTimer`,存入 `_name2timer`。**只能注册一次**(断言)。

**计时入口 `start_timer/stop_timer/insert_record`**:按 name 查 timer;`KeyError`(未注册)只 warn 一次并记入 `_unregistered_timer_*`;其他异常 catch 后只打日志——**绝不让插桩异常冒泡到训练**。`tag` 里若有 `NDTIMELINE_STREAM_KEY` 会取出作为 stream 传给 `start`。

**异步 flush `async_flush(...)`**(关键机制):
1. 拷贝所有 meta;若 `not keep_timer_state`,把副本的 `enabled` 设为 `next_iter_enabled`。
2. **换出**:把当前 `_name2timer` 作为 `past_name2timer`(带着已记录的 event),用 meta 副本新建一套 `fresh_name2timer`,然后在 `gc.disable()` 保护下把 `self._name2timer` 指向新字典。
   > 这一步是"记录与导出解耦"的核心:训练线程立即在新字典上继续 start/stop,旧字典(含未处理 event)交给后台慢慢同步。临时关 GC 是避免切换瞬间触发 CUDA event 析构。
3. 回收已完成 future(打印其中异常)。
4. 计算是否需要校准:`force_calibrate` 或(`dynamic_calibrate` 且距上次 >30min)。若 `sequential_calibrate`,现在就在主线程校准。
5. 若 `submit2handler` 且有可导出的 timer,把 `_flush_timers(handlers, past_name2timer, step_range, world_info, require_calibrate)` 提交到 `ThreadPoolExecutor`,记入 `_futures`。

**后台导出 `_flush_timers(...)`**:
1. 若 `require_calibrate`(非顺序校准情形)→ `GlobalReferenceTime.calibrate()`。
2. 逐个 timer:`elapsed_result = timer.elapsed()`;对每个 handler,按 `dispatch_mode`/`dst_names` 过滤后 `handler(name, *elapsed_result, step_range, world_info, extra)`(即展开成 `(name, elapsed, parts, since_parts, tags, step_range, world_info, extra)`)。handler 异常被 catch 打日志。
3. `timer.meta = None`(防循环引用导致 event GC 失败)。
4. 所有 timer 处理完后,给每个 handler 发一个特殊调用 `handler("special", 0.0, [], [], [], range(0,1), world_info, extra)`——作为"本批结束"的**分界信号**(`LocalTimelineNDHandler` 靠它判断何时落盘)。

**`wait()`**:有未完成 future 时,先 `dist.barrier()`,再 `wait(futures, timeout=10)`,收集异常,最后 `sleep(3)` 给 streamer 排空。

**单例**:`Singleton` metaclass + `NDTimerManagerSingleton`;`is_initialized()` 供所有 context manager 判断"没初始化就空转"。

### 4.6 NCCL stream 获取(`stream.py` + patched torch)

要量真实通信耗时,event 必须记到 NCCL 实际使用的 stream 上。PyTorch 原生不暴露该 stream,故依赖补丁新增的 C++ 接口:

- `getP2PCudaStreamId(device_idx, peer, is_batched)`:batched 按 device key 查 `ncclStreams_`,非 batched 按 `getKeySendRecv(rank, peer)` 查;命中返回 `stream.id()`,否则 `-1`。
- `getCollCudaStreamId(tensors)`:按 tensor 所在 device 的 key 查。

`get_nccl_p2p_stream`/`get_nccl_coll_stream` 把裸 id 包成 `torch.cuda.Stream`,按 metric 名缓存到全局 `NCCL_STREAMS`。

**懒分配 caveat**:NCCL stream 在第一次通信时才创建,所以第一次通信查不到(`stream_id<0`),首个通信 span 必然丢失,日志 `p2p stream is not available, skipped` 属预期。未打补丁的 torch 上 `hasattr(..., "get_p2p_cuda_stream_id")` 为假,通信 metric 全部退化为默认 stream 记录。

### 4.7 Streamer 子进程 + 二进制协议

**为什么要独立进程**:写文件/聚合是重活,放训练进程会抖动;且一台机多个 rank 各写各的文件不便对比。于是每台机由 `local_rank==0` 拉起一个 streamer 进程统一聚合。

**`NDtimelineStreamer.init(local_rank, handlers)`**:
- 非 `local_rank 0` 直接返回(每机一个)。
- 清理旧 socket,`torch.multiprocessing.spawn(start_method="spawn", daemon=True, join=False)` 起 1 个进程跑 `run`。

**`run(process_index, handlers)`**(在子进程内):
- 建 `queue.Queue(500000)`(在子进程里建,省主进程内存)。
- 起消费线程 `internal_queue_consume`:不断 `q.get()`,把每条消息展开调用各 post handler;handler 抛 `NDHandlerError` 只 warn 继续。
- 起 `ThreadingUnixStreamServer(SOCK_PATH, MsgHandler)` 监听 Unix socket。

**`MsgHandler.handle`**:每连接一线程,循环 `recv_and_validate(...)` 拆包 → `loads_fn`(pickle,前后关 GC)→ `q.put`。`BrokenPipeError`=客户端退出;`ProtocolValidationError`=坏包,跳过。

**二进制协议(`binary_protocol.py`)**:
```
| Magic 1B (0x9C) | Version 1B (0x00) | Reserved 2B ("00") |
| Payload Length 4B (little-endian, 上限 128MiB) |
| Payload (pickle protocol 4) |
| EOF 1B ('\n') |
```
`recv_and_validate` 逐字段校验,`read_or_recv`/`recv_to_buf` 用 `preload_data` 缓冲处理粘包/跨包残留。

**`SockNDHandler`**(训练进程侧客户端):`AF_UNIX` socket,`settimeout(2s)`;`try_to_connect` 带退避重试(`must=True` 时 50 次 ×0.8s),errno 106(已连接)视为成功;`call_impl` 把入参 dict 序列化 `sendall`。server 挂了置 `server_exited` 不再发——**采集失败绝不影响训练**。

### 4.8 Handler:step 归属 + 单机聚合 + Chrome Trace

**`parse_record(...)`**(把原始入参 → `DeviceTimerStreamRecord`):
- 校验 `recent_elapsed_raw_parts / since_parts / tags` 三者等长,否则 `NDHandlerError`。
- **两种 step 归属模式**:
  1. tag 里带 `_inner_global_step`(来自 `step_getter`):用 `itertools.groupby` 按 step 分组,每组一条 record(要求所有 tag 都带 step,否则报错)。
  2. 无 step tag:按 `step_range` **均分**——要求 span 数能被 step 数整除,否则报错并提示改用 `step_getter`。
- record 携带 rank/tp/pp/dp/ip/role/trial/run 等全维度信息("N-Dimension" 的由来)。

**`LocalTimelineNDHandler`**(聚合 + 落盘):
- 为每个 local rank 维护 `rank2buffer`(record 列表)和 `rank2steps`(deque of step-set,初始含 1 个空 set)。
- 普通 metric:`parse_record` 后 append 到对应 rank 的 buffer,并把 step 记入当前 set。
- 收到 `"special"`(flush 分界):给该 rank 的 deque append 新 set;**当所有 local rank 的 deque 长度 ≥2**,说明上一段收齐 → `dump_records()` 落盘,buffer 里只保留新段。
- `dump_records()`:
  - 遍历所有 rank 的 buffer,按 metric 名查 `thread_sort_index` 表(把 metric 映射到 `main` / `pp ->` / `pp <-` / `pp send/recv` / `tp collective` / `dp collective` / `other` 等线程类别);
  - `tid = rank*10 + sort_index`(乘 10 给类别编号让位),`pid = dp_rank`;
  - 每条 span → `CompleteEvent(ts=start_ts*1e6, dur=duration*1e6, ...)`(秒→微秒,Chrome Trace 单位);
  - 追加 `ThreadMetadataEvent` / `ProcessMetadataEvent`(命名线程/进程);
  - 所有 event `.to_objects()` 后 `json.dump` 到 `trace_step{min}_{max}`。
  - 切分缓冲时用 `record.copy()` 而非直接持引用,规避 GC 导致的内存泄漏。

**`chrome_trace_event.py`**:Chrome Trace Event Format 的数据类——`CompleteEvent`(ph=X)、`BeginEvent`(B)/`EndEvent`(E)、`FlowEvent`(s/t/f,画跨线程/进程连线,会自动修正"结束早于开始"以免不渲染)、`CounterEvent`(C)、`ProcessMetadataEvent`/`ThreadMetadataEvent`(M)。产出的 JSON 可直接被 Perfetto / `chrome://tracing` 打开。

### 4.9 级别过滤与 tag 白名单

- **级别**:`DeviceTimer.start/stop/insert_record` 入口判 `meta.level > level`(level 取自 manager 的 `metric_level`)则直接返回。低于阈值的 metric **连 event 都不记**,零开销。
- **tag 白名单**:只有 `meta.legal_tags` 里的键会保留进最终记录。注册时若给了 `step_getter`,会自动把 `_inner_global_step` 和 `stream_key` 加进白名单。

### 4.10 健壮性与资源管理细节

- 所有插桩入口、所有 handler 调用都 try/except,异常只记日志。
- 未注册 name 只 warn 一次(记入列表去重)。
- 多处 `gc.disable()` 保护:timer 字典切换、`pickle.loads`、`disable_and_save/recover_from_history`——避免 GC 触发 CUDA event 析构带来开销或失败。
- `timer.meta = None`、`record.copy()` 都是为了切断引用、帮助 event/record 正确回收。
- 双缓冲参考 event 让"校准"与"正常记录"互不踩踏。
- `wait()` 的 barrier + 超时 + sleep,保证退出时数据尽量不丢。

---

## 5. 局限与注意事项

1. **依赖 patched PyTorch 才能精确测通信**:无补丁时 `ndtimeit_p2p`/`ndtimeit_coll` 退化为默认 stream 计时(量的是发起时刻),并打印 skip 日志;计算计时不受影响。
2. **首个通信 span 必丢**:NCCL stream 懒分配所致,属预期。
3. **高层 `init_ndtimers` 绑定 veScale device mesh**:`hybrid` 模式需 `VESCALE_DEVICE_MESH` 已按 DP/TP/PP 初始化;`fsdp` 模式仅内部版。纯 PyTorch 请用低层 API(§3)。
4. **import 即依赖整个 legacy 包**:哪怕只用 ndtimeline,也要 `pip install -e .` 安装 legacy。
5. **socket 目录硬编码** `/opt/tiger/tmp/ndtimeline`,需可写;换目录要同步改 `variables.py` 的 `SOCK_PARENT_DIR`(streamer 与客户端都读它)。
6. **开源版只有单机聚合**:跨机器汇总需自实现一个 post handler(如 MQHandler)把各机 metric 送中心存储;内部版的 `MQNDHandler`/`SMon`/what-if 分析未开源。
7. **不自动告警**:ndtimeline 产出 trace 供人/下游分析,本身不做"最慢 N 个 rank"的自动判定与修复——那是 Megatron/NeMo StragglerDetector 或编排层的职责(详见 [`straggler-detection-survey.md`](./straggler-detection-survey.md))。
8. **全局时钟校准是集体操作**:`init`/`flush` 触发的校准含 barrier/all_gather,所有 rank 必须同步到达;不要只在部分 rank 调用 `flush`。
9. **`register_timers` 只能调一次**;step 用 `step_getter` 或 `inc_step/set_global_step`,二选一保持一致。
