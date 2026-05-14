

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

constexpr uint8_t kMortonX   = 1;  
constexpr uint8_t kMortonY   = 2;    
constexpr uint8_t kMortonZ   = 4;   
constexpr int kNumChildren   = 8;    
constexpr int kOccPad        = 256;  


struct Point {
    float x, y, z;
};

struct Node {
    int parent_index;                      
    int children_index[kNumChildren];    
    uint8_t occupancy;                    
    int depth;                            
    float center_x, center_y, center_z;   
    float extent;                          
    float z_extent;                        
    int start;                            
    int end;                              
    int cur_child_idx;                    
};

class Octree {
public:
    std::vector<Node> nodes;        
    std::vector<Point> points;      
    int max_depth;                   
    Octree(int max_d) : max_depth(max_d) {
        nodes.reserve(1 << 20);      
    }
};

void split_node(Octree& tree, int node_idx) {
    Node& node = tree.nodes[node_idx];
    int start = node.start;
    int end   = node.end;
    int num_pts = end - start;

    if (node.depth >= tree.max_depth || num_pts < 1)
        return;

    std::vector<Point> temp(tree.points.begin() + start, tree.points.begin() + end);
    std::vector<uint8_t> morton_codes(num_pts);
    int count[kNumChildren] = {0};

    for (int j = 0; j < num_pts; ++j) {
        uint8_t code = 0;
        if (temp[j].x > node.center_x) code |= kMortonX;
        if (temp[j].y > node.center_y) code |= kMortonY;
        if (temp[j].z > node.center_z) code |= kMortonZ;
        morton_codes[j] = code;
        count[code]++;
    }

    int offset[kNumChildren];
    offset[0] = start;
    for (int i = 1; i < kNumChildren; ++i)
        offset[i] = offset[i - 1] + count[i - 1];

    int cursor[kNumChildren];
    std::copy(offset, offset + kNumChildren, cursor);
    for (int j = 0; j < num_pts; ++j) {
        uint8_t code = morton_codes[j];
        tree.points[cursor[code]++] = temp[j];
    }

    node.occupancy = 0;
    for (int i = 0; i < kNumChildren; ++i) {
        int child_start = offset[i];
        int child_end   = (i == kNumChildren - 1) ? end : offset[i + 1];
        if (child_start == child_end) {
            node.children_index[i] = -1;
            continue;
        }

        float fx = (i & 1) ? 0.5f : -0.5f;
        float fy = (i & 2) ? 0.5f : -0.5f;
        float fz = (i & 4) ? 0.5f : -0.5f;

        Node child;
        child.parent_index   = node_idx;
        child.depth          = node.depth + 1;
        child.start          = child_start;
        child.end            = child_end;
        child.cur_child_idx  = i;
        child.center_x       = node.center_x + fx * node.extent;
        child.center_y       = node.center_y + fy * node.extent;
        child.center_z       = node.center_z + fz * node.z_extent;
        child.extent         = node.extent * 0.5f;
        child.z_extent       = node.z_extent * 0.5f;
        child.occupancy      = 0;
        std::fill(child.children_index, child.children_index + kNumChildren, -1);

        int child_idx = tree.nodes.size();
        tree.nodes.push_back(child);

        node.children_index[i] = child_idx;
        node.occupancy |= (1 << (7 - i));
    }
}


Octree build_octree(
    py::array_t<float, py::array::c_style | py::array::forcecast> points_np,
    int max_depth,
    std::vector<float> center,
    float extent,
    float z_extent,
    bool cylin = false
) {
    auto array_info = points_np.request();

    if (array_info.ndim != 2 || array_info.shape[1] != 3)
        throw std::runtime_error("points must be (N,3)");

    int N = array_info.shape[0];

    std::vector<Point> pts(N);
    std::memcpy(pts.data(), array_info.ptr, N * 3 * sizeof(float));

    Octree tree(max_depth);
    tree.points = std::move(pts);

    Node root;
    root.parent_index = -1;
    root.depth        = 1;
    root.start        = 0;
    root.end          = N;
    root.center_x     = center[0];
    root.center_y     = center[1];
    root.center_z     = center[2];
    root.extent       = extent;
    root.z_extent     = z_extent;
    root.cur_child_idx = 0;
    root.occupancy    = 0;
    std::fill(root.children_index, root.children_index + kNumChildren, -1);

    tree.nodes.push_back(root);

    float max_r = 0;
    if (cylin) {
        float max_x = -INFINITY;
        float max_y = -INFINITY;
        for (int i = 0; i < N; ++i) {
            if (tree.points[i].x > max_x) max_x = tree.points[i].x;
            if (tree.points[i].y > max_y) max_y = tree.points[i].y;
        }
        max_r = std::max(max_x, max_y);
    }

    std::queue<int> q;
    q.push(0);
    while (!q.empty()) {
        int node_idx = q.front();
        q.pop();
        Node& node = tree.nodes[node_idx];

        if (node.depth >= max_depth)
            continue;

        if (cylin) {
            float effective_radius = node.center_x;
            if ((effective_radius <= max_r/4 && node.depth >= max_depth - 2) ||
                (effective_radius > max_r/4 && effective_radius <= max_r/2 && node.depth >= max_depth - 1)) {
                continue;
            }
        }

        split_node(tree, node_idx);

        for (int i = 0; i < kNumChildren; ++i) {
            if (node.children_index[i] != -1)
                q.push(node.children_index[i]);
        }
    }
    return tree;
}

struct ContextResult {
    int N;                              
    int K;                              
    std::vector<int> Occupancy;         
    std::vector<int> Level;             
    std::vector<int> CurChildIdx;        
    std::vector<float> Pos;            
    std::vector<float> Extents;          
    std::vector<int> IsNonLeaf;         
};


ContextResult build_K_context(const Octree& tree, int K, bool include_leaves) {
    int total_N = tree.nodes.size();

    int valid_N;
    if (include_leaves) {
        valid_N = total_N;
    } else {
        valid_N = 0;
        for (const auto& node : tree.nodes) {
            if (node.occupancy != 0)
                valid_N++;
        }
    }

    std::vector<int> node_to_seq(total_N, -1);

    std::vector<int>    Occupancy(valid_N * K, kOccPad);
    std::vector<int>    Level(valid_N * K, 0);
    std::vector<int>    CurChildIdx(valid_N * K, 0);
    std::vector<float>  Pos(valid_N * K * 3, 0.0f);
    std::vector<float>  Extents(valid_N * K * 2, 0.0f);
    std::vector<int>    IsNonLeaf;
    if (include_leaves)
        IsNonLeaf.resize(valid_N);

    int seq_idx = 0;
    for (int i = 0; i < total_N; ++i) {
        const Node& node = tree.nodes[i];
        if (!include_leaves && node.occupancy == 0) continue;  

        node_to_seq[i] = seq_idx;

        if (include_leaves)
            IsNonLeaf[seq_idx] = (node.occupancy != 0) ? 1 : 0;

        if (node.parent_index >= 0 && node_to_seq[node.parent_index] != -1) {
            int parent_seq = node_to_seq[node.parent_index];
            std::memcpy(&Occupancy[seq_idx * K],
                        &Occupancy[parent_seq * K + 1],
                        (K - 1) * sizeof(int));
            std::memcpy(&Level[seq_idx * K],
                        &Level[parent_seq * K + 1],
                        (K - 1) * sizeof(int));
            std::memcpy(&CurChildIdx[seq_idx * K],
                        &CurChildIdx[parent_seq * K + 1],
                        (K - 1) * sizeof(int));
            std::memcpy(&Pos[seq_idx * K * 3],
                        &Pos[parent_seq * K * 3 + 3],
                        (K - 1) * 3 * sizeof(float));
            std::memcpy(&Extents[seq_idx * K * 2],
                        &Extents[parent_seq * K * 2 + 2],
                        (K - 1) * 2 * sizeof(float));
        }

        int last_slot = seq_idx * K + (K - 1);
        Level[last_slot]       = node.depth;
        CurChildIdx[last_slot] = node.cur_child_idx;
        Occupancy[last_slot]   = node.occupancy;  
        Pos[last_slot * 3 + 0] = node.center_x;
        Pos[last_slot * 3 + 1] = node.center_y;
        Pos[last_slot * 3 + 2] = node.center_z;
        Extents[last_slot * 2 + 0] = node.extent;
        Extents[last_slot * 2 + 1] = node.z_extent;

        seq_idx++;
    }
    return {valid_N, K, Occupancy, Level, CurChildIdx, Pos, Extents, IsNonLeaf};
}

std::vector<float> octree2pointcloud_flat(const Octree& tree) {
    std::vector<float> points;
    if (tree.nodes.empty()) return points;

    int leaf_count = 0;
    for (const auto& node : tree.nodes) {
        if (node.occupancy == 0)
            leaf_count++;
    }
    points.reserve(leaf_count * 3);

    for (const auto& node : tree.nodes) {
        if (node.occupancy == 0) {
            points.push_back(node.center_x);
            points.push_back(node.center_y);
            points.push_back(node.center_z);
        }
    }
    return points;
}
static_assert(sizeof(Point) == 3 * sizeof(float), "Point layout must match numpy (N,3) float32");
static_assert(std::is_trivially_copyable<Point>::value, "Point must be trivially copyable for memcpy");

PYBIND11_MODULE(octreeutils, m) {
    m.doc() = "";

    // Point — 三维点
    py::class_<Point>(m, "Point")
        .def(py::init<>())
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def_readwrite("z", &Point::z);

    // Node — 八叉树节点
    py::class_<Node>(m, "Node")
        .def_readonly("parent_index",  &Node::parent_index)
        .def_readonly("children_index", &Node::children_index)
        .def_readonly("occupancy",     &Node::occupancy)
        .def_readonly("depth",         &Node::depth)
        .def_readonly("center_x",      &Node::center_x)
        .def_readonly("center_y",      &Node::center_y)
        .def_readonly("center_z",      &Node::center_z)
        .def_readonly("extent",        &Node::extent)
        .def_readonly("z_extent",      &Node::z_extent)
        .def_readonly("cur_child_idx", &Node::cur_child_idx);

    py::class_<Octree>(m, "Octree")
        .def_readonly("nodes",     &Octree::nodes)
        .def_readonly("points",    &Octree::points)
        .def_readonly("max_depth", &Octree::max_depth);

    py::class_<ContextResult>(m, "ContextResult")
        .def_readonly("N",           &ContextResult::N)
        .def_readonly("K",           &ContextResult::K)
        .def_readonly("Occupancy",   &ContextResult::Occupancy)
        .def_readonly("Level",       &ContextResult::Level)
        .def_readonly("CurChildIdx", &ContextResult::CurChildIdx)
        .def_readonly("Pos",         &ContextResult::Pos)
        .def_readonly("Extents",     &ContextResult::Extents)
        .def_readonly("IsNonLeaf",   &ContextResult::IsNonLeaf);

    m.def("build_octree", &build_octree,
        py::arg("points"),
        py::arg("max_depth"),
        py::arg("center"),
        py::arg("extent"),
        py::arg("z_extent"),
        py::arg("cylin") = false,
        "BFS"
    );

    m.def("build_K_context", &build_K_context,
        py::arg("tree"),
        py::arg("K"),
        py::arg("include_leaves") = false,
        ""
    );

    m.def("octree2pointcloud_flat", &octree2pointcloud_flat,
        py::arg("tree"),
        ""
    );
}
