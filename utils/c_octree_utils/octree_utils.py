import numpy as np
import utils.c_octree_utils.octreeutils as octreeutils

def build_octree_py(
    points: np.ndarray,
    max_depth: int,
    center,
    extent,
    z_extent,
    cylin=False
):
    """
    points: (N, 3) float32
    """
    tree = octreeutils.build_octree(
        points,
        max_depth,
        list(center),
        float(extent),
        float(z_extent),
        cylin
    )
    return tree

def build_K_context_py(tree, K, include_leaves=False):
    ctx = octreeutils.build_K_context(tree, K, include_leaves)
    N = ctx.N
    K = ctx.K
    # -------- reshape --------
    Occupancy = np.array(ctx.Occupancy, dtype=np.int32).reshape(N, K)
    Level = np.array(ctx.Level, dtype=np.int32).reshape(N, K)
    CurChildIdx = np.array(ctx.CurChildIdx, dtype=np.int32).reshape(N, K)
    Pos = np.array(ctx.Pos, dtype=np.float32).reshape(N, K, 3)
    Extents = np.array(ctx.Extents, dtype=np.float32).reshape(N, K, 2)

    result = {
        "Occupancy": Occupancy,
        "Level": Level,
        "CurChildIdx": CurChildIdx,
        "Pos": Pos,
        "Extents": Extents,
    }
    if include_leaves:
        result["IsNonLeaf"] = np.array(ctx.IsNonLeaf, dtype=bool)
    return result

def octree2points_py(tree):
    pts = octreeutils.octree2pointcloud_flat(tree)
    return np.array(pts, dtype=np.float32).reshape(-1, 3)


def get_node_numlist(tree):
    """Return per-level node count (all nodes, leaf + non-leaf).

    Parameters
    ----------
    tree : Octree
        Octree object from :func:`build_octree_py`.

    Returns
    -------
    list[int]
        Number of nodes at each depth, indexed from depth 1 to max_depth.
    """
    depths = np.array([n.depth for n in tree.nodes], dtype=np.int32)
    return [int((depths == d).sum()) for d in range(1, tree.max_depth + 1)]

