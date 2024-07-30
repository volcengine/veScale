################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

RECV_FORWARD = "forward-recv"
RECV_BACKWARD = "backward-recv"
SEND_FORWARD = "forward-send"
SEND_BACKWARD = "backward-send"
SEND_FORWARD_RECV_BACKWARD = "forward-send-backward-recv"
SEND_BACKWARD_RECV_FORWARD = "backward-send-forward-recv"
CROSS_MESH_RECV = "cross-mesh-recv"
CROSS_MESH_SEND = "cross-mesh-send"
FORWARD_COMPUTE = "forward-compute"
BACKWARD_COMPUTE = "backward-compute"
UNSHARD_AG = "unshard-all-gather"
GRAD_RS = "grad-reduce-scatter"
GRAD_AR = "grad-all-reduce"
