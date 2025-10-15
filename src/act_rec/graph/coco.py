from __future__ import annotations

import numpy as np

from act_rec.graph import tools


NUM_JOINTS = 17

# COCO skeleton using the conventional keypoint order:
# 0 Nose, 1 Left Eye, 2 Right Eye, 3 Left Ear, 4 Right Ear,
# 5 Left Shoulder, 6 Right Shoulder, 7 Left Elbow, 8 Right Elbow,
# 9 Left Wrist, 10 Right Wrist, 11 Left Hip, 12 Right Hip,
# 13 Left Knee, 14 Right Knee, 15 Left Ankle, 16 Right Ankle
self_link = [(i, i) for i in range(NUM_JOINTS)]
inward_ori_index = [
    (1, 0),
    (2, 0),
    (3, 1),
    (4, 2),
    (5, 0),
    (6, 0),
    (7, 5),
    (8, 6),
    (9, 7),
    (10, 8),
    (11, 5),
    (12, 6),
    (13, 11),
    (14, 12),
    (15, 13),
    (16, 14),
]

inward = inward_ori_index
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode: str = "spatial", scale: int = 1):
        self.num_node = NUM_JOINTS
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_binary = tools.edge2mat(self.neighbor, self.num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + np.eye(self.num_node, dtype=np.float32))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

    def get_adjacency_matrix(self, labeling_mode: str | None = None) -> np.ndarray:
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            return tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        raise ValueError(f"Unsupported labeling_mode: {labeling_mode}")
