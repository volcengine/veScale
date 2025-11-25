# veScale Pipeline Parallel (PP)

## TLDR

<img src="../../docs/pictures/pp.png" alt="PP" width="700"/>

## What is PP?

`Pipeline Parallel` (`PP`) partitions layers of a model across multiple devices to form a pipelined execution of the training.
`PP` takes as input a list of microbatches of data per iteration and performs pipelined training execution (forward, backward, and optimizer update) on each microbatch, while overlaps communication with computation on each device.

## Why veScale PP?

Existing `PP` systems suffer multiple drawbacks as below, which prevent productization within a company:

- _Complex API_: assuming that model developers are also systems experts in `PP`

- _Hacking model code_: requiring manually rewrite the model code to run `PP`

- _Lacking single device abstraction_: requiring manually rewrite the training script to be `PP` device-specific

- _Lacking options of pipeline construction_: relying on a single option of graph tracing, or perfect graph tracing, or solely manual construction of the pipeline.

- _Lacking customizability of pipeline schedule_: deeply coupling the entire runtime (e.g., compute, communication) with a specific `PP` schedule (e.g., `1F1B`)

- _Lacking diverse model support_: supporting only sequential model architecture without branching, or supporting only pipeline stages having single input or single output without multiple input/output.

## What is veScale PP?

`veScale PP` offers a new `PP` framework that is both _**Easy-to-Use**_ and _**Easy-to-Customize**_, thus it is used internally in our production.
Especially, `veScale PP` provides:

- _Easy API_: hiding the complexity of `PP` systems and runtimes from model developers

- _Zero model code change_: keeping the original torch model code as it is for transparent pipelined models

- _Single device abstraction_: keeping the single device training script as it is for transparent pipelined training on multiple devices

- _Multiple options of pipeline construction_: user can flexibly choose modes:
    
    - `GRAPH_EAGER` mode automatically traces and parses the model into a graph, splits the graph into pipeline stages, and constructs each stage for pipeline execution
        
        - graph tracer can also be choices or users

    - `MANUAL_EAGER` mode manually constructs each pipeline stage for pipeline execution, without graph tracing, parsing, and splitting.

- _Customizable pipeline schedule_: empowering users to define their custom pipeline schedules, beyond our built-in schedule as below:
    
    - `1F1B`
    
    - `Interleaved 1F1B`

    - `Zero Bubble`

- _Support diverse models_: support comprehensive model archictures for non-sequential models, multiple-input-multiple-output stages, and etc. 

## Why is veScale PP a better option than its counterparts?

- Compared with Megatron-LM's PP, `veScale PP` offers not only a better __Ease-of-Use__ experience in all aspects (easy API, zero model code, single device abstraction, options of pipeline construction) but also a plus of __Customizability__ allowing users to conveniently customize new pipeline schedules.

- Compared with DeepSpeed, `veScale PP` requires no modification of model code. It further supports multi-stage scheduling for non-sequential multimodal architecture and multi-input settings instead of being constrained by `nn.Sequential`'s syntax.

- Compared with the pre-release torchtitan, `veScale PP` provides: i) single device abstraction of training script, ii) wider options of graph tracer support, iii) wider model architecture support, and iv) guarantees bitwise accuracy alignment between `PP` and single device code.

## How does veScale PP work?

Spinning up a `PP` job typically requires three steps: i) trace and parse model graph, ii) construct pipeline stage, and iii) execute pipeline schedule. Each step is handled by `PipeParser`, `PipeModule`, and `PipeEngine`. Upon receiving the model definition, `PipeParser` (`GRAPH_EAGER` mode) breaks down the model code to the intermediate representation of low-level modules and operators up to the granularity of your choice. Under `MANUAL_EAGER` mode, users only need to assign stage modules and their communication relationships.  `PipeModule` collects parameters and operators, and optimizer states belonging to the same stage, and resolves communication topology among devices. `PipeEngine` will schedule steps to execute training according to pipeline schedules.

## How to use veScale PP?

- Example of using `GRAPH_EAGER` mode:

    ```python
    # zero model code change 
    class EightMLP(nn.Module): 
       def __init__(self, ...):
           self.mlp1 = MLP(...)
           ...
           self.mlp8 = MLP(...)
       def forward(...):
           ...

    # An EightMLP is composed of 8 submodules called MLP
    model = EightMLP()
    # or model = deferred_init(EightMLP) 

    from vescale.plan import PipelineParallelPlan, PipelineScheduleType, PipelineSplitMethodType, ModeType
    from vescale.pipe import construct_pipeline_stage
    from vescale.engine import PipeEngine
    from vescale.dtensor.device_mesh import DeviceMesh
    
    # create 3-dim DeviceMesh
    device_mesh = DeviceMesh("cuda", [[[0]], [[1]], [[2]], [[3]]], mesh_dim_names=("PP", "DP", "TP"))
    
    # prepare plan for pipeline parallelism
    pipe_plan = PipelineParallelPlan(
        mode=ModeType.GRAPH_EAGER,
        split_method=PipelineSplitMethodType.MANUAL,
        num_stages=4,
        virtual_chunks=2,
        smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(8)],  # maintain hierarchy of each MLP module
        split_points=["mlp2", "mlp4", "mlp6", "mlp8"],  # managed pipeline split points by fully qualified names
        overlap_p2p_comm=True,  # speedup option
        schedule_type=PipelineScheduleType.INTERLEAVED_1F1B,
    )

    # parse model graph, split graph, and construct pipeline stage
    pipe_stage = construct_pipeline_stage(model, pipe_plan, device_mesh)

    # prepare pipeline schedule and execution engine
    engine = PipeEngine(pipe_stage, device_mesh, pipe_plan)

    # train PP model as if on single device
    for minibatch_data in dataloader:
        minibatch_loss, microbatch_outputs = engine(minibatch_data)
        minibatch_loss.backward()
        ...
    
    ```

- Example of using `MANUAL_EAGER` mode: Coming Soon.

- APIs can be found in `<repo>/legacy/vescale/pipe/pipe_stage.py` and `<repo>/vescale/pipe/pipe.py`

- More examples can be found in `<repo>/legacy/test/parallel/pipeline/api/test_simple_api.py`