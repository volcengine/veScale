################################################################################
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
# Modification Copyright 2024 ByteDance Ltd. and/or its affiliates.
################################################################################

import torch
import torch.nn as nn
import torch.fx as fx
import collections
import warnings
import math
import inspect
from torch.fx import Tracer, Graph, Proxy, GraphModule
from torch.fx.proxy import ParameterProxy
from transformers.utils.fx import (
    _proxies_to_metas,
    _generate_random_int,
    check_if_model_is_supported,
    _FX_SUPPORTED_MODELS_WITH_KV_CACHE,
    _IS_IN_DEBUG_MODE,
    _MANUAL_META_OVERRIDES,
    HFProxy,
    HFAttribute,
    HFTracer,
)

try:
    from transformers.utils.fx import _gen_constructor_wrapper
except Exception as e:
    warnings.warn("Util path changed. Now load from a new path")
    from transformers.utils.fx import gen_constructor_wrapper as _gen_constructor_wrapper

from transformers.utils.import_utils import (
    TORCH_FX_REQUIRED_VERSION,
    get_torch_version,
    is_torch_fx_available,
    is_peft_available,
)
from torch.fx._compatibility import compatibility
from transformers.modeling_utils import PreTrainedModel
from typing import Any, Callable, Dict, List, Optional, Union, Sequence, Type
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
)

if is_peft_available():
    from peft import PeftModel


_IS_PARTITION_MODULE = "PARTITION"


class ModelTracer(fx.Tracer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def is_leaf_module(self, m, module_qualified_name):
        return (
            m.__module__.startswith("torch.nn")
            or m.__module__.startswith("torch.ao.nn")
            or hasattr(m, _IS_PARTITION_MODULE)
        ) and not isinstance(m, torch.nn.Sequential)


class HFModelTracer(Tracer):
    """
    Tracer that is able to symbolically trace models from the library. To do that, it uses the HFProxy instead of the
    regular PyTorch torch.fx.Proxy.
    """

    # Feature flag for proxying accesses to buffer values
    proxy_buffer_attributes: bool = True
    allow_insert_stateless_mods: bool = True
    _TORCH_METHODS_TO_PATCH = [
        "arange",
        "zeros",
        "ones",
        "full",
        "full_like",
        "eye",
        "empty",
        "tensor",
        "clamp",
        "finfo",
    ]
    supported_archs = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)

    def __init__(self, autowrap_modules=(math,), autowrap_functions=(), partition_modules=None):
        super().__init__(autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions)

        if not is_torch_fx_available():
            raise ImportError(
                f"Found an incompatible version of torch. Found version {get_torch_version()}, but only version "
                f"{TORCH_FX_REQUIRED_VERSION} is supported."
            )

        self.visited_partition_module_paths = set()
        self.partition_module_classes_and_fqns = set() if partition_modules is None else set(partition_modules)

    def _generate_dummy_input(
        self, model: PreTrainedModel, input_name: str, shape: List[int], input_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Generates dummy input for model inference recording."""
        # Retrieving the model class, either from the "class_for_deserialization" attribute if the model was restored
        # from pickle, or from the "__class__" attribute in the general case.
        model_class_name = getattr(model, "class_for_deserialization", model.__class__).__name__
        device = model.device
        inputs_dict = {}

        # when tracing a model with KV cache, we simply need to unsure that the KV cache length is larger than one to
        # rightfully pass certain controlflows (Example: https://github.com/huggingface/transformers/blob/5c8d941d66734811d2ef6f57f15b44f7fb7a98c4/src/transformers/modeling_attn_mask_utils.py#L162).
        # After tracing, the model can then still be used with arbitrary lengths different than the one used during tracing.
        kv_cache_length = 5

        if input_name in ["labels", "start_positions", "end_positions"]:
            batch_size = shape[0]
            if model_class_name in [
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES),
                *get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
                *get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES),
            ]:
                inputs_dict["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class_name in [
                *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
                *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
                "XLNetForQuestionAnswering",
            ]:
                inputs_dict["start_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
                inputs_dict["end_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class_name in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
                if not hasattr(model.config, "problem_type") or model.config.problem_type is None:
                    raise ValueError(
                        "Could not retrieve the problem type for the sequence classification task, please set "
                        'model.config.problem_type to one of the following values: "regression", '
                        '"single_label_classification", or "multi_label_classification".'
                    )

                if model.config.problem_type == "regression":
                    labels_shape = (batch_size, model.config.num_labels)
                    labels_dtype = torch.float32
                elif model.config.problem_type == "single_label_classification":
                    labels_shape = (batch_size,)
                    labels_dtype = torch.long
                elif model.config.problem_type == "multi_label_classification":
                    labels_shape = (batch_size, model.config.num_labels)
                    labels_dtype = torch.float32
                else:
                    raise ValueError(
                        'Expected model.config.problem_type to be either: "regression", "single_label_classification"'
                        f', or "multi_label_classification", but "{model.config.problem_type}" was provided.'
                    )
                inputs_dict["labels"] = torch.zeros(*labels_shape, dtype=labels_dtype, device=device)

            elif model_class_name in [
                *get_values(MODEL_FOR_PRETRAINING_MAPPING_NAMES),
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES),
                "GPT2DoubleHeadsModel",
                "PeftModelForCausalLM",
                "PeftModelForSeq2SeqLM",
            ]:
                inputs_dict["labels"] = torch.zeros(shape, dtype=torch.long, device=device)
            elif model_class_name in [*get_values(MODEL_FOR_CTC_MAPPING_NAMES)]:
                inputs_dict["labels"] = torch.zeros(shape, dtype=torch.float32, device=device)
            else:
                raise NotImplementedError(
                    f"Generating the dummy input named {input_name} for {model_class_name} is not supported yet."
                )
        elif "pixel_values" in input_name:
            batch_size = shape[0]
            image_size = getattr(model.config, "image_size", None)
            if image_size is None:
                if hasattr(model.config, "vision_config"):
                    image_size = model.config.vision_config.image_size
                elif hasattr(model.config, "encoder"):
                    image_size = model.config.encoder.image_size
                else:
                    image_size = (_generate_random_int(), _generate_random_int())

            # If no num_channels is in the config, use some arbitrary value.
            num_channels = getattr(model.config, "num_channels", 3)
            if not isinstance(image_size, collections.abc.Iterable):
                image_size = (image_size, image_size)
            height, width = image_size
            inputs_dict[input_name] = torch.zeros(
                batch_size, num_channels, height, width, dtype=torch.float32, device=device
            )
        elif "bbox" in input_name:
            inputs_dict[input_name] = torch.zeros(*shape, 4, dtype=torch.float, device=device)
        elif "input_features" in input_name:
            inputs_dict[input_name] = torch.zeros(
                *shape, model.config.input_feat_per_channel, dtype=torch.float, device=device
            )
        elif "visual_feats" in input_name:
            inputs_dict[input_name] = torch.zeros(
                shape
                + [
                    model.config.visual_feat_dim,
                ],
                dtype=torch.float,
                device=device,
            )
        elif "visual_pos" in input_name:
            inputs_dict[input_name] = torch.zeros(
                shape
                + [
                    model.config.visual_pos_dim,
                ],
                dtype=torch.float,
                device=device,
            )
        elif "inputs" in input_name:
            inputs_dict[input_name] = torch.zeros(*shape, dtype=torch.float, device=device)
        elif "input_values" in input_name:
            batch_size, _ = shape
            # Generating big sequence length for audio inputs.
            seq_length = _generate_random_int(low=10000, high=20000)
            inputs_dict[input_name] = torch.zeros(batch_size, seq_length, dtype=torch.float, device=device)
        elif "mask" in input_name:
            if "past_key_values" in input_names:
                mask_shape = [shape[0], shape[1] + kv_cache_length]
            else:
                mask_shape = shape

            inputs_dict[input_name] = torch.zeros(mask_shape, dtype=torch.long, device=device)
        elif "ids" in input_name:
            inputs_dict[input_name] = torch.zeros(shape, dtype=torch.long, device=device)
        elif "past_key_values" in input_name:
            if model.config.model_type not in _FX_SUPPORTED_MODELS_WITH_KV_CACHE:
                raise NotImplementedError(
                    f"Symbolic trace with past_key_values input is not supported yet for the model {model.config.model_type}. Please open an issue or a PR in Transformers repository if you would like to see the support added."
                )
            num_heads = model.config.num_attention_heads
            head_dim = model.config.hidden_size // model.config.num_attention_heads

            cache_shape = (shape[0], num_heads, kv_cache_length, head_dim)
            pkv = tuple(
                (
                    torch.rand(cache_shape, dtype=torch.float, device=device),
                    torch.rand(cache_shape, dtype=torch.float, device=device),
                )
                for i in range(model.config.num_hidden_layers)
            )
            inputs_dict[input_name] = pkv
        else:
            shape_with_hidden_size = shape + [model.config.hidden_size]
            inputs_dict[input_name] = torch.zeros(shape_with_hidden_size, dtype=torch.float, device=device)

        return inputs_dict

    def is_leaf_module(self, m, module_qualified_name):
        return (not self._stateless_mod_instanciation_depends_on_proxies(m)) and (
            hasattr(m, _IS_PARTITION_MODULE)
            or (
                m.__module__.startswith("torch.nn")
                or m.__module__.startswith("torch.ao.nn")
                and not isinstance(m, torch.nn.Sequential)
            )
        )

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

        if kind == "placeholder" and target in self.meta_args:
            rv.install_metadata(self.meta_args[target])
            return rv

        if target in self.orig_fns:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs:
                kwargs["device"] = "meta"

        try:
            args_metas = torch.fx.node.map_aggregate(args, _proxies_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, _proxies_to_metas)

            if kind == "call_function":
                meta_target = _MANUAL_META_OVERRIDES.get(target, target)
                meta_out = meta_target(*args_metas, **kwargs_metas)
                if isinstance(meta_out, torch.Tensor):
                    meta_out = meta_out.to(device="meta")
            elif kind == "call_method":
                method = getattr(args_metas[0].__class__, target)
                meta_target = _MANUAL_META_OVERRIDES.get(method, method)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_module":
                if not hasattr(self, "orig_forward"):
                    raise AttributeError(f"{self} does not have an attribute called orig_forward")
                self._disable_module_getattr = True
                try:
                    mod = self.root.get_submodule(target)

                    mod_type = type(mod)
                    assert not any(path for path in self.visited_partition_module_paths if target.startswith(path))
                    self.visited_partition_module_paths.add(target)
                    # assert mod_type not in self.partition_module_classes
                    if mod_type in _MANUAL_META_OVERRIDES:
                        meta_out = _MANUAL_META_OVERRIDES[mod_type](mod, *args_metas, **kwargs_metas)
                    else:
                        if self.partition_module_classes_and_fqns and (
                            target in self.partition_module_classes_and_fqns
                            or mod_type in self.partition_module_classes_and_fqns
                        ):
                            raise ValueError  # not to recurse into partition module's forward()
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                except:  # noqa: E722
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    meta_out = None
                finally:
                    self._disable_module_getattr = False
            elif kind == "get_attr":
                self._disable_module_getattr = True
                try:
                    attr_itr = self.root
                    atoms = target.split(".")
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    if isinstance(attr_itr, torch.Tensor):
                        meta_out = attr_itr.to(device="meta")
                    else:
                        meta_out = attr_itr
                finally:
                    self._disable_module_getattr = False
            else:
                return rv

            if not isinstance(rv, Proxy):
                raise ValueError("Don't support composite output yet")
            rv.install_metadata(meta_out)
        except Exception as e:
            if _IS_IN_DEBUG_MODE:
                warnings.warn(f"Could not compute metadata for {kind} target {target}: {e}")

        return rv

    # Replaced by .getattr from PyTorch 1.13
    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, "_disable_module_getattr", False):
            return attr_val
        else:

            def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
                for n, p in collection_to_search:
                    if attr_val is p:
                        if n not in parameter_proxy_cache:
                            kwargs = {}
                            if "proxy_factory_fn" in inspect.signature(self.create_proxy).parameters:
                                kwargs["proxy_factory_fn"] = (
                                    None
                                    if not self.param_shapes_constant
                                    else lambda node: ParameterProxy(self, node, n, attr_val)
                                )
                            val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                            parameter_proxy_cache[n] = val_proxy
                        return parameter_proxy_cache[n]
                return None

            if isinstance(attr_val, torch.nn.Parameter):
                maybe_parameter_proxy = maybe_get_proxy_for_attr(
                    attr_val, self.root.named_parameters(), parameter_proxy_cache
                )
                if maybe_parameter_proxy is not None:
                    return maybe_parameter_proxy

            if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
                maybe_buffer_proxy = maybe_get_proxy_for_attr(
                    attr_val, self.root.named_buffers(), parameter_proxy_cache
                )
                if maybe_buffer_proxy is not None:
                    return maybe_buffer_proxy

            return attr_val

    # Needed for PyTorch 1.13+
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        return self._module_getattr(attr, attr_val, parameter_proxy_cache)

    def call_module(self, m, forward, args, kwargs):
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    def proxy(self, node):
        return HFProxy(node, self)

    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
        dummy_inputs: Optional[Dict[str, Any]] = None,
        complete_concrete_args_with_inputs_not_in_dummy_inputs: bool = True,
    ) -> Graph:
        """
        Traces `root` and returns the corresponding FX `torch.fx.Graph` representation. `root` can either be a
        `torch.nn.Module` instance or a Python callable. Note that after this call, `self.root` may be different from
        the `root` passed in here. For example, when a free function is passed to `trace()`, we will create a
        `torch.nn.Module` instance to use as the root and add embedded constants to.

        Args:
            root (`torch.nn.Module` or  `Callable`):
                Either a `torch.nn.Module`` or a function to be traced through. If root is not a
                [`~transformers.PreTrainedModel`], then `dummy_inputs` must be passed, otherwise tracing will fail.
            concrete_args (`Dict[str, Any], *optional*):
                Concrete arguments that should not be treated as Proxies
            dummy_inputs (`Dict[str, Any]`, *optional*):
                The dummy inputs needed to handle data-dependent control-flow if `root` is not a
                [`~transformers.PreTrainedModel`]. It can also be used when `root` is a
                [`~transformers.PreTrainedModel`] to specify custom dummy inputs for a subset or all the model inputs.
            complete_concrete_args_with_inputs_not_in_dummy_inputs (`bool`, *optional*, defaults to `True`):
                If `True`, and `dummy_inputs` is specified, every argument that `root` can take that is not in
                `dummy_inputs` and not in `concrete_args` will be added to `concrete_args`, otherwise does nothing.

        Returns:
            `torch.fx.Graph`:
                A FX `torch.fx.Graph` representing the semantics of the passed-in `root`.

        """
        sig = inspect.signature(root.forward if isinstance(root, torch.nn.Module) else root)

        if concrete_args is None:
            concrete_args = {}

        if dummy_inputs is not None and complete_concrete_args_with_inputs_not_in_dummy_inputs:
            for param in sig.parameters.values():
                if param.name in dummy_inputs:
                    continue
                if param.default is inspect.Parameter.empty:
                    raise ValueError(f"You need to specify a default value for the parameter {param.name}.")
            concrete_args.update(
                {
                    p.name: p.default
                    for p in sig.parameters.values()
                    if (p.name not in dummy_inputs and p.name not in concrete_args)
                }
            )

        input_names = sig.parameters.keys() - concrete_args.keys()

        # Creating a random input shape to generate dummy inputs.
        batch_size = _generate_random_int()
        sequence_length = _generate_random_int()
        shape = [batch_size, sequence_length]

        if root.__class__.__name__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES):
            num_choices = _generate_random_int(low=2, high=5)
            shape.insert(1, num_choices)

        inputs = dict(dummy_inputs) if dummy_inputs is not None else {}
        for input_name in input_names:
            if input_name in inputs:
                continue
            # We enforce that root must either be a PreTrainedModel or deserialized from a serialized traced model to
            # be able to use HFTracer._generate_dummy_input.
            if isinstance(root, self.supported_archs) or type(root).__qualname__.startswith(
                ("_deserialize_graph_module", "_CodeOnlyModule")
            ):
                inputs.update(self._generate_dummy_input(root, input_name, shape, input_names=input_names))
            else:
                raise RuntimeError(
                    f"Could not generate input named {input_name} for because root is not a"
                    " transformers.PreTrainedModel."
                )

        concrete_metas = {
            input_name: input_.to("meta") if isinstance(input_, torch.Tensor) else input_
            for input_name, input_ in inputs.items()
        }
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD and param.name not in input_names:
                concrete_metas[f"**{param.name}"] = {}
        self.meta_args = concrete_metas
        self.patched_torch_methods = {
            target: _gen_constructor_wrapper(getattr(torch, target)) for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            self.graph = super().trace(root, concrete_args=concrete_args)
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)

        # This is necessary because concrete args are added as input to the traced module since
        # https://github.com/pytorch/pytorch/pull/55888.
        for node in self.graph.nodes:
            if node.op == "placeholder":
                # Removing default values for inputs as the forward pass will fail with them.
                if node.target in input_names:
                    node.args = ()
                    # Without this, torch.jit.script fails because the inputs type is Optional[torch.Tensor].
                    # It cannot infer on the attributes and methods the input should have, and fails.
                    node.type = torch.Tensor
                # It is a concrete arg so it is not used and should be removed.
                else:
                    to_visit = [node]
                    to_delete = collections.OrderedDict()
                    while to_visit:
                        n = to_visit.pop(0)
                        to_delete[n] = None
                        to_visit += list(n.users.keys())

                    for user in reversed(to_delete.keys()):
                        self.graph.erase_node(user)

            # Without this, return type annotation "Tuple" is causing code execution failure.
            if node.op == "output":
                node.type = None

        return self.graph

    def _stateless_mod_instanciation_depends_on_proxies(self, mod: nn.Module) -> bool:
        """
        Whether the module was instantiated with Proxies. If that is the case, such module cannot be a leaf module
        because its attributes are input-dependent.
        """
        return any(isinstance(attr, Proxy) for attr in mod.__dict__.values())

    def _insert_module_as_submodule(self, mod: nn.Module) -> str:
        """
        Helper method which tries to insert a module that was not declared as submodule.
        """
        # If one of the module attributes is a Proxy, it means that its instantiation is input-dependent.
        # It is not possible to insert such modules, those should be traced through.
        if self._stateless_mod_instanciation_depends_on_proxies(mod):
            return ""
        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        already_inserted = False
        while hasattr(self.root, path):
            if getattr(self.root, path) is mod:
                already_inserted = True
                break
            path = f"{mod_name}_{idx}"
            idx += 1

        # No need to add multiple instances of the same module.
        if not already_inserted:
            self.root.add_module(path, mod)
        return path

    def path_of_module(self, mod: nn.Module) -> str:
        """
        Helper method to find the qualified name of `mod` in the Module hierarchy of `root`. For example, if `root` has
        a submodule named `foo`, which has a submodule named `bar`, passing `bar` into this function will return the
        string "foo.bar".

        Args:
            mod (str): The `Module` to retrieve the qualified name for.
        """
        try:
            return super().path_of_module(mod)
        except NameError as e:
            if self.allow_insert_stateless_mods and len(list(mod.parameters())) == 0 and len(list(mod.buffers())) == 0:
                path = self._insert_module_as_submodule(mod)
                return path
            raise e

    @compatibility(is_backward_compatible=True)
    def keys(self, obj: "Proxy") -> Any:
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an iterator if ** is supposed to work in
        your custom tracer.
        """
        attribute = HFAttribute(obj, "keys")()
        if obj.node.target == "**kwargs":
            return attribute._metadata
        return attribute


def get_concrete_args(model: nn.Module, input_names: List[str]):
    sig = inspect.signature(model.forward)

    if not (set(input_names) <= set(sig.parameters.keys())):
        formatted_input_names = input_names[0] if len(input_names) == 1 else ", ".join(input_names)
        formatted_allowed_input_names = ", ".join(sig.parameters.keys())
        raise ValueError(
            f"The model does not have input(s) named: {formatted_input_names}, expected a subset of the following:"
            f" {formatted_allowed_input_names}"
        )

    return {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}


def hf_symbolic_trace(
    model: PreTrainedModel,
    input_names: Optional[List[str]] = None,
    disable_check: bool = False,
    tracer_cls: Type[HFTracer] = HFTracer,
    partition_modules: List = None,
) -> GraphModule:
    """
    Performs symbolic tracing on the model.

    Args:
        model ([`PretrainedModel`]):
            The model to trace.
        input_names (`List[str]`, *optional*):
            The names of the inputs of the traced model. If unset, model.dummy_inputs.keys() are used instead.
        disable_check (`bool`, *optional*, defaults to `False`):
            If `True`, no check is done before trying to trace the model, this is mostly usesul for debugging purposes.
        tracer_cls (`Type[HFTracer]`, *optional*, defaults to `HFTracer`):
            The tracer class to use for instantiating the tracer. If unset, `HFTracer` is used instead.
        partition_modules (`List`):
            A list of string paths to un-partitionable submodules of custom modulen classes.

    Returns:
        `torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.

    Example:

        ```python
        from transformers.utils.fx import symbolic_trace

        traced_model = symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
        ```
    """
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    input_names = list(input_names)
    concrete_args = get_concrete_args(model, input_names)

    if not disable_check:
        check_if_model_is_supported(model)

    # Tracing.
    if partition_modules:
        # annotate partition modules as minimally unpartitionable units in stage split.
        assert isinstance(partition_modules, Sequence)

        def _check_legitimate_fqn(unique_paths, path):
            return not any(path == p or path.startswith(p + ".") for p in unique_paths)

        partition_modules_paths = set()
        for fqn, sub_module in model.named_modules():
            if (fqn in partition_modules) or (
                type(sub_module) in partition_modules and _check_legitimate_fqn(partition_modules_paths, fqn)
            ):
                # elif type(sub_module) in partition_modules and _check_legitimate_fqn(partition_modules_paths, fqn):
                partition_modules_paths.add(fqn)
        partition_modules_paths = list(partition_modules_paths)
        register_partition_module(model, fully_qualified_names=partition_modules_paths)
    tracer = tracer_cls(partition_modules=partition_modules)
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)

    if hasattr(model, "config"):
        traced.config = model.config
    # The model class must be stored as an attribute to allow model deserialization, which uses trace, and thus
    # _generate_dummy_input, where the model class is needed.
    traced.class_for_deserialization = model.__class__
    if hasattr(model, "device"):
        traced.device = model.device

    return traced


def register_partition_module(module: nn.Module, fully_qualified_names: Union[str, Sequence] = None):
    if fully_qualified_names is None:
        setattr(module, _IS_PARTITION_MODULE, True)
    else:
        if isinstance(fully_qualified_names, str):
            fully_qualified_names = [fully_qualified_names]
        for fqn, sub_module in module.named_modules():
            for mod_name in fully_qualified_names:
                if fqn == mod_name:
                    setattr(sub_module, _IS_PARTITION_MODULE, True)
