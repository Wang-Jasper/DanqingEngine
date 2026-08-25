#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "physics/Shape.h"

// DynamicAABBTree — general dynamic AABB tree: broadphase + ray queries over
// leaf userData. Node pool + free list, O(log n) insert/remove, Box2D-style
// subtree heuristic; leaves are opaque int userData.

class DynamicAABBTree
{
public:
    static constexpr int NullNode = -1;

    DynamicAABBTree();

    // Insert a leaf; returns proxyId (for update/remove)
    int insert(const AABB &aabb, int userData);

    // Update a leaf AABB (skip if still inside the old AABB, else remove + reinsert)
    void update(int proxyId, const AABB &aabb);

    // Remove a leaf
    void remove(int proxyId);

    // AABB query: calls visitor(userData) for every leaf overlapping aabb
    void queryAABB(const AABB &aabb, const std::function<void(int)> &visitor) const;

    // Ray query: a visitor returning false stops traversal early
    //   from / dir: ray origin / direction (dir need not be normalized)
    //   maxT: max distance multiple along dir (hit_point = from + t * dir, t <= maxT)
    void queryRay(const glm::vec3 &from, const glm::vec3 &dir, float maxT,
                  const std::function<bool(int)> &visitor) const;

    // Frustum query: calls visitor(userData) for leaves conservatively intersecting the frustum.
    //   planes: 6 planes; n = (a,b,c) points inside the frustum, n*p + d >= 0 is inside;
    //   order irrelevant but usually Left/Right/Bottom/Top/Near/Far (aligned with utils/Frustum.h::Frustum)
    // Uses the p-vertex test: per plane, take the corner of node.aabb most toward the plane's
    //   positive side; if that corner is outside (n*p + d < 0) the whole subtree is culled.
    struct QueryPlane
    {
        glm::vec3 n;
        float d;
    };
    void queryFrustum(const QueryPlane planes[6],
                      const std::function<void(int)> &visitor) const;

    // Self-collision traversal: calls visitor(userDataA, userDataB) for every
    // overlapping leaf pair, once each (pair ordering semantics left to the visitor)
    void selfOverlap(const std::function<void(int, int)> &visitor) const;

    int nodeCount() const { return static_cast<int>(nodes_.size()) - freeCount_; }
    int rootIndex() const { return root_; }
    bool empty() const { return root_ == NullNode; }
    void clear();

private:
    struct Node
    {
        AABB aabb;
        int parent = NullNode;
        int left = NullNode;
        int right = NullNode;
        int height = -1; // -1 means in the free list; leaves are 0
        int userData = -1;

        bool isLeaf() const { return left == NullNode; }
    };

    // Node pool management
    int allocateNode();
    void freeNode(int id);

    // Insert/remove leaves into the tree structure
    void insertLeaf(int leaf);
    void removeLeaf(int leaf);

    // Walk up fixing aabb + height, attempting a rotation balance at each node
    int balance(int iA);

    // Merge two AABBs
    static AABB unionAABB(const AABB &a, const AABB &b);
    static float surfaceArea(const AABB &a);
    static bool overlap(const AABB &a, const AABB &b);

    std::vector<Node> nodes_;
    int root_ = NullNode;
    int freeList_ = NullNode; // head of the free-node list
    int freeCount_ = 0;

    // ---------------------------------------------------------------------
    // Scratch stacks reused by query* / selfOverlap to avoid per-call heap allocs.
    // ---------------------------------------------------------------------
    // mutable: queries are const on the tree but need writable stacks.
    // queryAABB / queryRay share scratchQueryStack_ (single-node stack);
    // selfOverlap additionally needs scratchPairStack_ + scratchSingleStack_.
    // Convention: clear() then push_back() at function entry.
    mutable std::vector<int> scratchQueryStack_;
    mutable std::vector<int> scratchSingleStack_;
    struct PairFrame
    {
        int a;
        int b;
    };
    mutable std::vector<PairFrame> scratchPairStack_;
};
