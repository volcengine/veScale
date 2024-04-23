// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "deferred_init.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "fake.h"
#include "stack_utils.h"

namespace torchdistx {

using at::DispatchKey;
using at::DispatchKeySet;
using at::FunctionSchema;
using at::irange;
using at::IValue;
using at::nullopt;
using at::OperatorHandle;
using at::optional;
using at::Storage;
using at::Tensor;
using at::TensorBase;
using at::TensorList;
using at::ThreadLocalState;
using at::ThreadLocalStateGuard;

using at::impl::GetVariableHooks;
using at::impl::SetVariableHooks;
using at::impl::VariableHooksInterface;

using c10::impl::tls_is_dispatch_key_excluded;
using c10::impl::tls_is_dispatch_key_included;

using torch::jit::Stack;

}  // namespace torchdistx

namespace torchdistx::detail {
namespace {

IValue copyIValue(const IValue& src) {
  IValue::HashAliasedIValueMap memo{};

  auto visitor = [&memo](const IValue& v) {
    // Deep-copy the compound objects and shallow-copy the rest.
    if (!v.isTuple() && !v.isList() && !v.isGenericDict()) {
      memo[v] = v;
    }

    return false;
  };

  src.visit(visitor);

  return src.deepcopy(memo);
}

// Creates a copy of `src` by deep-copying all its compound objects (i.e. lists,
// tuples, and dictionaries).
Stack copyStack(const Stack& src) {
  Stack dst{};

  dst.reserve(src.size());

  for (auto i : irange(src.size())) {
    const IValue& value = torch::jit::peek(src, i, src.size());

    torch::jit::push_one(dst, copyIValue(value));
  }

  return dst;
}

class OpNode;

// Describes a particular operation output including its node in the operation
// graph and its output index.
class OpOutputDescriptor {
 public:
  explicit OpOutputDescriptor(std::shared_ptr<OpNode> node, std::size_t output_index) noexcept
      : node_{std::move(node)}, output_index_{output_index} {}

  const std::shared_ptr<OpNode>& node() const noexcept {
    return node_;
  }

  std::size_t output_index() const noexcept {
    return output_index_;
  }

 private:
  std::shared_ptr<OpNode> node_;
  std::size_t output_index_;
};

// Each fake tensor constructed in a deferred-init context has its individual
// instance of `TensorRecord` stored along with the tensor.
class TensorRecord {
 public:
  const OpOutputDescriptor& output_descriptor() const {
    return opt_output_desc_.value();
  }

  void set_output_descriptor(OpOutputDescriptor&& output_desc) noexcept {
    opt_output_desc_ = std::move(output_desc);
  }

  // Forces the record instance of `view` to be kept alive even if `view` goes
  // goes out of scope. This is necessary when `view` is a view of the current
  // tensor and has an in-place operation. In such case we have to ensure that
  // we don't delete recorded operations that are only referenced by `view`.
  void keepAlive(const Tensor& view);

 private:
  optional<OpOutputDescriptor> opt_output_desc_{};
  std::vector<std::shared_ptr<TensorRecord>> view_records_{};
};

void TensorRecord::keepAlive(const Tensor& view) {
  auto record = unsafeAsFake(view).getData<TensorRecord>(DispatchKey::DeferredInit);

  TORCH_INTERNAL_ASSERT(record,
      "The tensor has no recorded deferred-init operation.");

  view_records_.emplace_back(std::move(record));
}

// An operation recorded in a deferred-init context.
class Op {
 public:
  using OpFn = std::function<void(Stack&)>;

 public:
  explicit Op(std::string name, OpFn fn, std::size_t num_args, std::size_t num_outputs, Stack s);

 public:
  static Op fromOperatorHandle(const OperatorHandle& handle, Stack s);

  const std::string& name() const noexcept {
    return name_;
  }

  bool materialized() const noexcept {
    return materialized_;
  }

  void materialize();
  void materializeWithShape(c10::IntArrayRef shape, const c10::optional<c10::Device> device);

  std::size_t num_outputs() const noexcept {
    return num_outputs_;
  }

  // This function can only be called after the operation is materialized.
  const Tensor& getOutput(std::size_t idx) const noexcept;

  void processTensorArguments(const TensorProcessor& processor) const {
    processTensors(stack_, num_args_, processor);
  }

  void convertTensorArguments(const TensorConverter& converter) {
    convertTensors(stack_, num_args_, converter);
  }

 private:
  void validateStack(const Stack& s) const;

 private:
  std::string name_;
  OpFn fn_;
  std::size_t num_args_;
  std::size_t num_outputs_;
  Stack stack_;
  optional<ThreadLocalState> tls_{};
  bool materialized_ = false;
};

Op::Op(std::string name, OpFn fn, std::size_t num_args, std::size_t num_outputs, Stack s)
    : name_{std::move(name)},
      fn_{std::move(fn)},
      num_args_{num_args},
      num_outputs_{num_outputs},
      stack_(std::move(s)) {
  // Capture the local thread state by the time of the operation.
  tls_ = ThreadLocalState{};

  validateStack(stack_);
}

Op Op::fromOperatorHandle(const OperatorHandle& handle, Stack s) {
  auto fn = [&handle](Stack& st) {
    handle.callBoxed(st);
  };

  const FunctionSchema& shm = handle.schema();
  return Op{shm.name(), std::move(fn), shm.arguments().size(), shm.returns().size(), std::move(s)};
}

void Op::validateStack(const Stack& s) const {
  // We only allow immutable types in the stack since otherwise we cannot
  // guarantee that we will have the same state during materialization.
  auto visitor = [this](const IValue& value) {
    TORCH_CHECK(value.isBool() ||
                value.isComplexDouble() ||
                value.isDevice() ||
                value.isDouble() ||
                value.isEnum() ||
                value.isGenerator() ||
                value.isGenericDict() ||
                value.isInt() ||
                value.isList() ||
                value.isNone() ||
                value.isString() ||
                value.isTensor() ||
                value.isTuple() ||
                value.isSymInt(),
          "`", name_, "` has an argument of type `", value.type()->str(), "` which is not "
          "supported in a deferred-init context.");

    return false;
  };

  for (auto i : irange(s.size())) {
    torch::jit::peek(s, i, s.size()).visit(visitor);
  }
}

void Op::materialize() {
  if (materialized_) {
    return;
  }

  {
    ThreadLocalStateGuard state_guard{*tls_};

    fn_(stack_);
  }

  fn_ = nullptr;

  tls_ = nullopt;

  materialized_ = true;
}

void Op::materializeWithShape(c10::IntArrayRef shape, const c10::optional<c10::Device> device) {
  if (materialized_) {
    return;
  }

  {
    ThreadLocalStateGuard state_guard{*tls_};

    auto replace_first_shape = [&](c10::IntArrayRef sp){
      IValue local_shape(sp);
      stack_[0] = local_shape; 
    };

    std::vector<std::string> op_white_list{"aten::randn", "aten::rand", "aten::empty", "aten::ones", "aten::zeros", "aten::full" };

    if (std::find(op_white_list.begin(),op_white_list.end(), name()) != op_white_list.end()){
      // if the op is operator
      replace_first_shape(shape);
    }

    if(device.has_value()){ // set target device 
      for (size_t i = 0 ; i < stack_.size(); i++){
        if(stack_[i].isDevice()){
          stack_[i] = IValue(device.value());
        }
      }
    }

    fn_(stack_);
  }

  fn_ = nullptr;

  tls_ = nullopt;

  materialized_ = true;
}

const Tensor& Op::getOutput(std::size_t idx) const noexcept {
  const Tensor* opt_out = nullptr;

  std::size_t i = 0;

  // Technically an operation can return arbitrary compound objects with mixed
  // types. This means we cannot directly index the output and have to perform
  // a linear search. Since most operations have only one or a small number of
  // outputs this isn't a big concern though.
  auto fn = [&opt_out, &idx, &i](const Tensor& tensor) {
    if (idx == i) {
      opt_out = &tensor;

      return true;
    } else {
      i++;
    }

    return false;
  };

  processTensors(stack_, num_outputs_, fn);

  TORCH_INTERNAL_ASSERT(opt_out != nullptr,
      "'", name_, "' has no tensor output at index ", idx , ".");

  return *opt_out;
}

inline TensorRecord& getTensorRecord(const Tensor& fake) {
  auto* record = unsafeAsFake(fake).unsafeGetData<TensorRecord>(DispatchKey::DeferredInit);

  TORCH_INTERNAL_ASSERT(record != nullptr,
      "The tensor has no recorded deferred-init operation.");

  return *record;
}

// A node in the operation graph holding a recorded operation.
class OpNode {
 public:
  explicit OpNode(std::uint64_t op_nr, Op&& op, const Stack& outputs);

  OpNode(const OpNode&) = delete;

  OpNode& operator=(const OpNode&) = delete;

  OpNode(OpNode&&) = delete;

  OpNode& operator=(OpNode&&) = delete;

  ~OpNode();

 private:
  void recordStorages(const Stack& outputs);

  void ensureViewsKeptAlive(const Stack& outputs);

  void ensureViewsKeptAlive(const Stack& outputs, const Tensor& fake_argument);

  void attachDependencies();

  void detachDependencies() noexcept;

 public:
  const Op& op() noexcept {
    return op_;
  }

  // Materializes the operation held by this node along with all the operations
  // in its recorded call stack.
  void materialize();
  // with changed shape
  void materializeWithShape(c10::IntArrayRef shape, c10::optional<c10::Device> device);

 private:
  void buildCallStack();

  class WalkContext {
   public:
    explicit WalkContext(const Storage& storage) noexcept : storage_{&storage} {}

    bool hasVisited(const OpNode* node);

    const Storage& storage() const noexcept {
      return *storage_;
    }

   private:
    const Storage* storage_{};
    std::unordered_set<const OpNode*> visited_{};
  };

  // Returns the node of the last in-place operation performed on the output
  // tensors of this operation.
  OpNode* getLastInPlaceOpNode();

  OpNode* getLastInPlaceOpNode(WalkContext& ctx);

  // Collects all operations callable from this node up until `last_node`.
  void collectCallStack(OpNode* last_node, std::vector<OpNode*>& out);

  void collectCallStack(OpNode* last_node, std::vector<OpNode*>& out, WalkContext& ctx);

  // Indicates whether any output tensors of this operation uses `storage`.
  bool usesStorage(const Storage& storage) const noexcept;

  void materializeArguments();

 private:
  // The chronological order of the operation held by this node.
  std::uint64_t op_nr_;
  // The operation held by this node.
  Op op_;
  // The `Storage` instances of the operation's tensor outputs recorded in the
  // deferred-init context. They are used to determine in-place operations.
  std::vector<Storage> storages_{};
  // The operation output descriptors that return the tensors used as inputs in
  // this node's operation.
  std::vector<OpOutputDescriptor> dependencies_{};
  // For tensor inputs constructed outside of the deferred-init context, their
  // version counters at the time of the recording. These counters are used to
  // verify that there have been no in-place updates to such tensors.
  std::vector<std::int64_t> argument_versions_{};
  // The nodes holding the operations that depend on this node's operation to
  // populate their input tensors.
  std::unordered_set<OpNode*> dependents_{};
  // The call stack of this operation; only populated during a materialization
  // call.
  std::vector<OpNode*> call_stack_{};
};

OpNode::OpNode(std::uint64_t op_nr, Op&& op, const Stack& outputs)
    : op_nr_{op_nr}, op_{std::move(op)} {
  recordStorages(outputs);

  ensureViewsKeptAlive(outputs);

  attachDependencies();
}

OpNode::~OpNode() {
  detachDependencies();
}

void OpNode::recordStorages(const Stack& outputs) {
  auto fn = [this](const Tensor& tensor) {
    // Ignore tensors that are not constructed in a deferred-init context since
    // we don't need to materialize them.
    if (isFake(tensor)) {
      storages_.emplace_back(unsafeAsFake(tensor).meta_storage());
    }

    return false;
  };

  processTensors(outputs, op_.num_outputs(), fn);
}

void OpNode::ensureViewsKeptAlive(const Stack& outputs) {
  auto fn = [this, &outputs](const Tensor& argument) {
    if (isFake(argument)) {
      ensureViewsKeptAlive(outputs, argument);
    }

    return false;
  };

  op_.processTensorArguments(fn);
}

void OpNode::ensureViewsKeptAlive(const Stack& outputs, const Tensor& fake_argument) {
  const Storage& fake_argument_storage = unsafeAsFake(fake_argument).meta_storage();

  auto fn = [&fake_argument, &fake_argument_storage](const Tensor& output) {
    // Check if the output is a view of the argument meaning they are different
    // tensors but share the same storage.
    if (isFake(output) && !output.is_same(fake_argument)) {
      if (unsafeAsFake(output).meta_storage().is_alias_of(fake_argument_storage)) {
        // Since the output is a view of the argument we have to ensure that the
        // operation node of the output stays alive even after all references to
        // the output get released. Otherwise we can't correctly materialize the
        // node of the argument.
        getTensorRecord(fake_argument).keepAlive(output);
      }
    }
    return false;
  };

  processTensors(outputs, op_.num_outputs(), fn);
}

void OpNode::attachDependencies() {
  auto fn = [this](Tensor& argument) {
    // If `argument` was constructed in the deferred-init context, add its node
    // to the dependencies.
    if (isFake(argument)) {
      TensorRecord& record = getTensorRecord(argument);

      const OpOutputDescriptor& dependency = record.output_descriptor();

      dependencies_.emplace_back(dependency);

      // Have a weak reference from the dependency to this node. This will be
      // used to resolve in-place operations.
      dependency.node()->dependents_.emplace(this);

      // Release the fake argument to avoid reference cycles.
      argument = Tensor{};
    } else {
      // Otherwise if we have a real tensor, record its version counter. This
      // information will be used to verify that it has the same state during
      // materialization.
      if (argument.is_inference()) {
        argument_versions_.emplace_back(0);
      } else {
        argument_versions_.emplace_back(argument._version());
      }
    }

    return false;
  };

  op_.convertTensorArguments(fn);
}

void OpNode::detachDependencies() noexcept {
  for (auto& dependency : dependencies_) {
    dependency.node()->dependents_.erase(this);
  }

  dependencies_.clear();
}

void OpNode::materialize() {
  // Do not try to shortcut this function by checking if the node is already
  // materialized. A later in-place operation can still change the output of
  // this node.

  buildCallStack();

  for (OpNode* node : call_stack_) {
    if (node->op_.materialized()) {
      continue;
    }

    node->materializeArguments();

    node->op_.materialize();

    // Make sure that we deallocate parts of the operation graph that are not
    // needed anymore.
    node->detachDependencies();
  }

  call_stack_.clear();
}

void OpNode::materializeWithShape(c10::IntArrayRef shape, const c10::optional<c10::Device> device) {
  // Do not try to shortcut this function by checking if the node is already
  // materialized. A later in-place operation can still change the output of
  // this node.

  buildCallStack();

  for (OpNode* node : call_stack_) {
    if (node->op_.materialized()) {
      continue;
    }

    node->materializeArguments();

    node->op_.materializeWithShape(shape, device);

    // Make sure that we deallocate parts of the operation graph that are not
    // needed anymore.
    node->detachDependencies();
  }

  call_stack_.clear();
}

void OpNode::buildCallStack() {
  OpNode* last_node = getLastInPlaceOpNode();

  collectCallStack(last_node, call_stack_);

  // Sort the operations by their chronological order.
  std::sort(call_stack_.begin(), call_stack_.end(), [](OpNode* lhs, OpNode* rhs) {
    return lhs->op_nr_ < rhs->op_nr_;
  });
}

OpNode* OpNode::getLastInPlaceOpNode() {
  OpNode* last_node = nullptr;

  for (const Storage& storage : storages_) {
    WalkContext ctx{storage};

    OpNode* node = getLastInPlaceOpNode(ctx);
    if (last_node == nullptr || node->op_nr_ > last_node->op_nr_) {
      last_node = node;
    }
  }

  return last_node;
}

OpNode* OpNode::getLastInPlaceOpNode(WalkContext& ctx) {
  if (ctx.hasVisited(this) || !usesStorage(ctx.storage())) {
    return nullptr;
  }

  OpNode* last_node = nullptr;

  // No need to search dependencies since their operation numbers can never be
  // greater than this node's operation number.
  for (OpNode* dependent : dependents_) {
    OpNode* node = dependent->getLastInPlaceOpNode(ctx);
    if (node != nullptr) {
      if (last_node == nullptr || node->op_nr_ > last_node->op_nr_) {
        last_node = node;
      }
    }
  }

  if (last_node == nullptr) {
    return this;
  } else {
    return last_node;
  }
}

void OpNode::collectCallStack(OpNode* last_node, std::vector<OpNode*>& out) {
  for (const Storage& storage : storages_) {
    WalkContext ctx{storage};

    collectCallStack(last_node, out, ctx);
  }
}

void OpNode::collectCallStack(OpNode* last_node, std::vector<OpNode*>& out, WalkContext& ctx) {
  if (ctx.hasVisited(this)) {
    return;
  }

  // All nodes that chronologically come before this node should be included in
  // the call stack.
  for (const auto& dependency : dependencies_) {
    dependency.node()->collectCallStack(last_node, out, ctx);
  }

  // If we have an in-place operation, check dependent nodes as well.
  if (usesStorage(ctx.storage())) {
    for (OpNode* dependent : dependents_) {
      // If the dependent node chronologically comes later than `last_node`, we
      // should skip it.
      if (dependent->op_nr_ > last_node->op_nr_) {
        continue;
      }

      // If the dependent node has an in-place operation as well, collect its
      // call stack since its output will affect this node's output.
      if (dependent->usesStorage(ctx.storage())) {
        dependent->collectCallStack(last_node, out, ctx);
      } else {
        // Otherwise we have to materialize the dependent node because its input
        // from this node will be modified in-place by a later operation.
        dependent->collectCallStack(dependent, out);
      }
    }
  }

  out.emplace_back(this);
}

bool OpNode::WalkContext::hasVisited(const OpNode* node) {
  if (visited_.find(node) == visited_.end()) {
    visited_.emplace(node);

    return false;
  } else {
    return true;
  }
}

bool OpNode::usesStorage(const Storage& storage) const noexcept {
  return std::any_of(storages_.begin(), storages_.end(), [&storage](const auto& s) {
    return storage.is_alias_of(s);
  });
}

void OpNode::materializeArguments() {
  auto dep_pos = dependencies_.begin();

  auto arg_ver_pos = argument_versions_.begin();

  auto fn = [this, &dep_pos, &arg_ver_pos](Tensor& argument) {
    if (argument.defined()) {
      TORCH_CHECK(!argument.is_inference(),
          "A `Tensor` argument required for the materialization of `", op_.name(), "` was created "
          "in inference mode. Materialization cannot be performed because in-place updates to "
          "inference tensors cannot be tracked.");

      TORCH_CHECK(argument._version() == *arg_ver_pos,
          "A `Tensor` argument required for the materialization of `", op_.name(), "` was updated "
          "in-place. Materialization cannot be performed.");

      ++arg_ver_pos;
    } else {
      const OpOutputDescriptor& dependency = *dep_pos;

      argument = dependency.node()->op_.getOutput(dependency.output_index());

      ++dep_pos;
    }
  };

  op_.convertTensorArguments(fn);
}

void ensureTensorRecordSet(const Op& op, Stack& outputs);

// Used to maintain the chronological order of operations.
thread_local std::uint64_t op_nr_ = 0;

void recordOp(Op&& op, Stack& outputs) {
  ensureTensorRecordSet(op, outputs);

  auto node = std::make_shared<OpNode>(op_nr_++, std::move(op), outputs);

  std::size_t idx = 0;

  // Associate every tensor returned by the operation with a descriptor that
  // holds the graph node and the output index. This information is used for
  // incrementally building the operation graph and for materalization.
  auto fn = [&node, &idx](Tensor& tensor) {
    if (isFake(tensor)) {
      OpOutputDescriptor output_desc{node, idx};

      getTensorRecord(tensor).set_output_descriptor(std::move(output_desc));
    }

    idx++;

    return false;
  };

  convertTensors(outputs, node->op().num_outputs(), fn);
}

void ensureTensorRecordSet(const Op& op, Stack& outputs) {
  auto fn = [](Tensor& tensor) {
    if (isFake(tensor)) {
      if (FakeTensor fake = unsafeAsFake(tensor); !fake.hasData(DispatchKey::DeferredInit)) {
        fake.setData(DispatchKey::DeferredInit, std::make_shared<TensorRecord>());
      }
    }

    return false;
  };

  convertTensors(outputs, op.num_outputs(), fn);
}

Tensor materialize(const Tensor& fake) {
  TensorRecord& record = getTensorRecord(fake);

  const OpOutputDescriptor& output_desc = record.output_descriptor();

  output_desc.node()->materialize();

  Tensor out = output_desc.node()->op().getOutput(output_desc.output_index());

  // Unfortunately there is no way for us to track calls to `requires_grad_()`,
  // so instead we explicitly set `requires_grad` after materialization.
  if (fake.is_leaf() && fake.requires_grad()) {
    out.set_requires_grad(true);
  }

  return out;
}

Tensor materialize_with_shape(const Tensor& fake, c10::IntArrayRef shape, const c10::optional<c10::Device> device) {
  TensorRecord& record = getTensorRecord(fake);

  const OpOutputDescriptor& output_desc = record.output_descriptor();

  output_desc.node()->materializeWithShape(shape, device);

  Tensor out = output_desc.node()->op().getOutput(output_desc.output_index());

  // Unfortunately there is no way for us to track calls to `requires_grad_()`,
  // so instead we explicitly set `requires_grad` after materialization.
  if (fake.is_leaf() && fake.requires_grad()) {
    out.set_requires_grad(true);
  }

  return out;
}

// The catch-all handler for the `DeferredInit` dispatch key.
class DeferredInitHandler {
 public:
  explicit DeferredInitHandler(const OperatorHandle& op, DispatchKeySet key_set, Stack* s) noexcept
      : handle_{&op}, key_set_{key_set}, stack_{s} {}

  void run();

 private:
  void validateTensorArguments() const;

  // Indicates whether an operation requires non-fake arguments to compute its
  // output (e.g. `aten::item()`).
  bool isTerminalOp() const noexcept;

  void materializeFakeArguments();

  bool hasFakeArgument() const noexcept;

  void redispatchToFake();

  bool hasFakeOutput() const noexcept;

  bool hasFakeTensorInStack(std::size_t n) const noexcept;

 private:
  static const DispatchKeySet kAfterDeferredInitKeySet_;

  const OperatorHandle* handle_;
  DispatchKeySet key_set_;
  Stack* stack_;
};

// NOLINTNEXTLINE(cert-err58-cpp)
const DispatchKeySet DeferredInitHandler::kAfterDeferredInitKeySet_{DispatchKeySet::FULL_AFTER,
                                                                    DispatchKey::DeferredInit};

void DeferredInitHandler::run() {
  NoDeferredInit guard{};

  validateTensorArguments();

  // An operation such as a call to `aten::item()` is considered terminal since
  // it requires non-fake arguments to compute its output.
  if (isTerminalOp()) {
    materializeFakeArguments();

    // None of the arguments are fake at this point, so the `Fake` handler will
    // transparently forward the operation to the real backend.
    redispatchToFake();
  } else {
    bool has_fake_arg = hasFakeArgument();

    // Preserve the original call frame before it gets overriden by the output
    // value(s) of the operation.
    Stack original_stack = copyStack(*stack_);

    redispatchToFake();

    if (has_fake_arg || hasFakeOutput()) {
      // Preserve the operator handle, the thread local state, and a copy of the
      // call frame. We need them later to materialize the operation.
      Op op = Op::fromOperatorHandle(*handle_, std::move(original_stack));

      recordOp(std::move(op), *stack_);
    }
  }
}

void DeferredInitHandler::validateTensorArguments() const {
  // If a tensor is fake, we expect it to be constructed in a deferred-init context.
  auto fn = [this](const Tensor& tensor) {
    TORCH_CHECK_VALUE(!isFake(tensor) || unsafeAsFake(tensor).hasData(DispatchKey::DeferredInit),
        "`", handle_->schema().name(), "` has a fake `Tensor` argument which was not constructed "
        "in a deferred-init context.");

    return false;
  };

  processTensors(*stack_, handle_->schema().arguments().size(), fn);
}

inline bool DeferredInitHandler::isTerminalOp() const noexcept {
  return handle_->schema().name() == "aten::item";
}

void DeferredInitHandler::materializeFakeArguments() {
  auto fn = [](Tensor& tensor) {
    if (isFake(tensor)) {
      tensor = materialize(tensor);
    }
  };

  convertTensors(*stack_, handle_->schema().arguments().size(), fn);
}

inline bool DeferredInitHandler::hasFakeArgument() const noexcept {
  return hasFakeTensorInStack(handle_->schema().arguments().size());
}

void DeferredInitHandler::redispatchToFake() {
  // The `Fake` handler will force newly-constructed tensors to be fake.
  key_set_ = key_set_.add(DispatchKey::Fake);

  handle_->redispatchBoxed(key_set_ & kAfterDeferredInitKeySet_, stack_);
}

inline bool DeferredInitHandler::hasFakeOutput() const noexcept {
  return hasFakeTensorInStack(handle_->schema().returns().size());
}

bool DeferredInitHandler::hasFakeTensorInStack(std::size_t n) const noexcept {
  bool has_fake = false;

  auto fn = [&has_fake](const Tensor& tensor) {
    if (isFake(tensor)) {
      has_fake = true;

      return true;
    } else {
      return false;
    }
  };

  processTensors(*stack_, n, fn);

  return has_fake;
}

void runDeferredInitHandler(const OperatorHandle& op, DispatchKeySet key_set, Stack* s) {
  DeferredInitHandler{op, key_set, s}.run();
}

void enableDeferredInitHandler(bool value) noexcept {
  c10::impl::tls_set_dispatch_key_included(DispatchKey::DeferredInit, value);
}

bool isDeferredInitEnabled() noexcept {
  if (tls_is_dispatch_key_included(DispatchKey::DeferredInit)) {
    return !tls_is_dispatch_key_excluded(DispatchKey::DeferredInit);
  } else {
    return false;
  }
}

}  // namespace
}  // namespace torchdistx::detail

// NOLINTNEXTLINE(cert-err58-cpp, clang-diagnostic-reserved-identifier)
TORCH_LIBRARY_IMPL(_, DeferredInit, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&torchdistx::detail::runDeferredInitHandler>());
}

namespace torchdistx {
namespace detail {
namespace {

void runGetVariableData(Stack& s) {
  IValue self = torch::jit::pop(s);

  TensorBase data = self.toTensor().variable_data();

  torch::jit::push(s, std::move(data));
}

void runSetVariableData(Stack& s) {
  IValue data = torch::jit::pop(s);
  IValue self = torch::jit::pop(s);

  self.toTensor().set_data(data.toTensor());

  torch::jit::push(s, std::move(self));
}

// Records calls to `Tensor::variable_data()`.
void recordGetVariableData(const TensorBase& self, const TensorBase& data) {
  if (!isFake(self) || !isFake(data)) {
    return;
  }

  IValue self_v{Tensor{self}};
  IValue data_v{Tensor{data}};

  Stack inp{};
  Stack out{};

  torch::jit::push(inp, self_v);
  torch::jit::push(out, data_v);

  constexpr const char* op_name = "VariableHooks::variable_data";

  std::size_t num_args = inp.size();

  recordOp(Op{op_name, runGetVariableData, num_args, out.size(), std::move(inp)}, out);
}

// Records calls to `Tensor::set_data()`.
void recordSetVariableData(const TensorBase& self, const TensorBase& data) {
  if (!isFake(self) || !isFake(data)) {
    return;
  }

  IValue self_v{Tensor{self}};
  IValue data_v{Tensor{data}};

  Stack inp{};
  Stack out{};

  torch::jit::push(inp, self_v, data_v);
  torch::jit::push(out, self_v);

  constexpr const char* op_name = "VariableHooks::set_data";

  std::size_t num_args = inp.size();

  recordOp(Op{op_name, runSetVariableData, num_args, out.size(), std::move(inp)}, out);
}

using AutogradBackwardHook = std::function<TensorBase(const TensorBase&)>;

// To record calls to the `VariableHooks` interface the deferred-init context
// uses an additional mechanism besides its dispatch handler. It replaces the
// global `VariableHooks` instance with a proxy that records the calls before
// forwarding them.
class ProxyVariableHooks : public VariableHooksInterface {
 public:
  explicit ProxyVariableHooks(VariableHooksInterface* inner) noexcept : inner_{inner} {}

  ProxyVariableHooks(const ProxyVariableHooks&) = delete;

  ProxyVariableHooks& operator=(const ProxyVariableHooks&) = delete;

  ProxyVariableHooks(ProxyVariableHooks&&) = delete;

  ProxyVariableHooks& operator=(ProxyVariableHooks&&) = delete;

  ~ProxyVariableHooks() override = default;

  TensorBase tensor_data(const TensorBase& self) const override {
    return inner_->tensor_data(self);
  }

  TensorBase variable_data(const TensorBase& self) const override;

  const std::shared_ptr<torch::autograd::Node>& grad_fn(const TensorBase& self) const override {
    return inner_->grad_fn(self);
  }

  unsigned int _register_hook(const TensorBase& self, AutogradBackwardHook hook) const override {
    return inner_->_register_hook(self, std::move(hook));
  }

  void remove_hook(const TensorBase& self, unsigned int pos) const override {
    return inner_->remove_hook(self, pos);
  }

  bool is_view(const TensorBase& self) const override {
    return inner_->is_view(self);
  }

  const TensorBase& base(const TensorBase& self) const override {
    return inner_->base(self);
  }

  const std::string& name(const TensorBase& self) const override {
    return inner_->name(self);
  }

  bool is_leaf(const TensorBase& self) const override {
    return inner_->is_leaf(self);
  }

  std::int64_t output_nr(const TensorBase& self) const override {
    return inner_->output_nr(self);
  }

  void set_data(const TensorBase& self, const TensorBase& data) const override;

  TensorBase data(const TensorBase& self) const override {
    return inner_->data(self);
  }

  std::int64_t _version(const TensorBase& self) const override {
    return inner_->_version(self);
  }

  void retain_grad(const TensorBase& self) const override {
    inner_->retain_grad(self);
  }

  bool retains_grad(const TensorBase& self) const override {
    return inner_->retains_grad(self);
  }

  void _backward(const Tensor& self, TensorList inputs, const optional<Tensor>& gradient,
                 optional<bool> keep_graph, bool create_graph) const override {
    inner_->_backward(self, inputs, gradient, keep_graph, create_graph);
  }

  void requires_grad_(const TensorBase& self, bool value) const override {
    inner_->requires_grad_(self, value);
  }

  void basic_autograd_not_implemented_fallback(const c10::OperatorHandle& op,
                                               c10::DispatchKeySet dispatch_keys,
                                               torch::jit::Stack* stack) const override {
    inner_->basic_autograd_not_implemented_fallback(op, dispatch_keys, stack);
  }

  VariableHooksInterface* inner() noexcept {
    return inner_;
  }

 private:
  static void validateTensorArgument(const char* op_name, const TensorBase& tensor) {
    TORCH_CHECK_VALUE(!isFake(tensor) || unsafeAsFake(tensor).hasData(DispatchKey::DeferredInit),
        "`VariableHooks::", op_name, "` has a fake `Tensor` argument which was not constructed in "
        "a deferred-init context.");
  }

 private:
  VariableHooksInterface* inner_;
};

TensorBase ProxyVariableHooks::variable_data(const TensorBase& self) const {
  if (isDeferredInitEnabled()) {
    validateTensorArgument("variable_data", self);
  }

  TensorBase data = inner_->variable_data(self);

  if (isDeferredInitEnabled()) {
    recordGetVariableData(self, data);
  }

  return data;
}

void ProxyVariableHooks::set_data(const TensorBase& self, const TensorBase& data) const {
  if (isDeferredInitEnabled()) {
    validateTensorArgument("set_data", self);
    validateTensorArgument("set_data", data);

    recordSetVariableData(self, data);
  }

  inner_->set_data(self, data);
}

class ProxyVariableHooksHolder {
 public:
  // Replaces Autograd's global `VariableHooks` instance with a proxy instance
  // that records hook function calls to the operation graph before forwarding
  // them to Autograd.
  void replaceGlobalHooks();

  void restoreGlobalHooks() noexcept;

 private:
  std::mutex mutex_{};
  std::unique_ptr<ProxyVariableHooks> hooks_{};
  std::size_t hooks_ref_count_ = 0;
};

void ProxyVariableHooksHolder::replaceGlobalHooks() {
  std::lock_guard<std::mutex> guard{mutex_};

  hooks_ref_count_++;

  if (hooks_ref_count_ == 1) {
    VariableHooksInterface* inner = GetVariableHooks();

    hooks_ = std::make_unique<ProxyVariableHooks>(inner);

    SetVariableHooks(hooks_.get());
  }
}

void ProxyVariableHooksHolder::restoreGlobalHooks() noexcept {
  std::lock_guard<std::mutex> guard{mutex_};

  if (hooks_ref_count_ == 0) {
    return;
  }

  hooks_ref_count_--;

  if (hooks_ref_count_ == 0) {
    SetVariableHooks(hooks_->inner());

    hooks_ = nullptr;
  }
}

ProxyVariableHooksHolder variable_hooks_holder{};

void replaceVariableHooks() {
  variable_hooks_holder.replaceGlobalHooks();
}

void restoreVariableHooks() noexcept {
  variable_hooks_holder.restoreGlobalHooks();
}

}  // namespace
}  // namespace detail

namespace {

thread_local std::size_t tls_deferred_init_level = 0;

}  // namespace

void enterDeferredInit() {
  tls_deferred_init_level++;

  if (tls_deferred_init_level == 1) {
    detail::enableDeferredInitHandler(true);

    detail::replaceVariableHooks();
  }
}

void leaveDeferredInit() noexcept {
  if (tls_deferred_init_level == 0) {
    return;
  }

  tls_deferred_init_level--;

  if (tls_deferred_init_level == 0) {
    detail::enableDeferredInitHandler(false);

    detail::restoreVariableHooks();
  }
}

bool canMaterialize(const Tensor& tensor) noexcept {
  return isFake(tensor) && unsafeAsFake(tensor).hasData(DispatchKey::DeferredInit);
}


Tensor materializeTensor(const Tensor& tensor) {
  if (canMaterialize(tensor)) {
    return detail::materialize(tensor);
  } else {
    return tensor;
  }
}

Tensor materializeTensorWithLocalShape(const at::Tensor& tensor, c10::IntArrayRef shape, const c10::optional<c10::Device> device){
  if (canMaterialize(tensor)) {
    return detail::materialize_with_shape(tensor, shape, device);
  } else {
    return tensor;
  }
}

bool isGenByRandomOp(const Tensor& tensor) noexcept{
  if (canMaterialize(tensor)) {
    detail::TensorRecord& record = detail::getTensorRecord(tensor);
    const detail::OpOutputDescriptor& output_desc = record.output_descriptor();
    auto name = output_desc.node()->op().name();
    std::vector<std::string> op_white_list{"aten::randn", "aten::rand"};
    return std::find(op_white_list.begin(),op_white_list.end(), name) != op_white_list.end();
  }else{
    return false;
  }
}

}  // namespace torchdistx
