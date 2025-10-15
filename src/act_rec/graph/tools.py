from __future__ import annotations

import numpy as np


def edge2mat(link: list[tuple[int, int]], num_node: int) -> np.ndarray:
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in link:
        A[j, i] = 1.0
    return A


def get_adjacency_matrix(edges: list[tuple[int, int]], num_nodes: int) -> np.ndarray:
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        A[edge] = 1.0
    return A


def normalize_digraph(A: np.ndarray) -> np.ndarray:
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node), dtype=np.float32)
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return (A @ Dn).astype(np.float32)


def normalize_adjacency_matrix(A: np.ndarray) -> np.ndarray:
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5, where=node_degrees > 0)
    norm_degs_matrix = np.eye(len(node_degrees), dtype=np.float32) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def get_k_scale_graph(scale: int, A: np.ndarray) -> np.ndarray:
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0], dtype=np.float32)
    for _ in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1.0
    return An.astype(np.float32)


def get_spatial_graph(
    num_node: int,
    self_link: list[tuple[int, int]],
    inward: list[tuple[int, int]],
    outward: list[tuple[int, int]],
) -> np.ndarray:
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    return np.stack((I, In, Out))
