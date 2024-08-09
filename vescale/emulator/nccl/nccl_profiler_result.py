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


from typing import List
from vescale.emulator.nccl.include.graph import NcclTopoGraph
import xml.etree.ElementTree as ET
from vescale.emulator.nccl.constants import *  # noqa: F403


def get_default_min_max_compcap():
    return 80, 80


def parse_graph_xml(xmlfile: str) -> List[NcclTopoGraph]:
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    graphs = []
    for graph in root.findall("graph"):
        pattern = int(graph.get("pattern"))
        nChannels = int(graph.get("nchannels"))
        bwIntra = float(graph.get("speedintra"))
        bwInter = float(graph.get("speedinter"))
        latencyInter = float(graph.get("latencyinter"))
        typeIntra = graph.get("typeintra")
        # Convert typeIntra string to an appropriate integer or keep as string based on your requirements
        typeIntra = LINK_LOC if typeIntra == "LOC" else LINK_NVL if typeIntra == "NVL" else -1  # Example conversion
        sameChannels = int(graph.get("samechannels"))

        topo_graph = NcclTopoGraph(
            pattern=pattern,
            nChannels=nChannels,
            bwIntra=bwIntra,
            bwInter=bwInter,
            latencyInter=latencyInter,
            typeIntra=typeIntra,
            sameChannels=sameChannels,
        )

        graphs.append(topo_graph)

    return graphs


def parse_nccl_topo(pg):
    xmlfile = pg.get_nccl_graph_xml()

    graphs = parse_graph_xml(xmlfile)
    ringgraph = graphs[NCCL_ALGO_RING]
    treegraph = graphs[NCCL_ALGO_TREE]
    nchannels = min(ringgraph.nChannels, treegraph.nChannels)
    # ncclTopoPostset
    nchannels = min(MAXCHANNELS, nchannels * 2)
    minCompCap, maxCompCap = get_default_min_max_compcap()
    return graphs, nchannels, minCompCap, maxCompCap


if __name__ == "__main__":
    pass
