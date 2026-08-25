#include "physics/DynamicAABBTree.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// AABB helpers
AABB DynamicAABBTree::unionAABB(const AABB &a, const AABB &b)
{
    AABB r;
    r.min = glm::min(a.min, b.min);
    r.max = glm::max(a.max, b.max);
    return r;
}

float DynamicAABBTree::surfaceArea(const AABB &a)
{
    glm::vec3 d = a.max - a.min;
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

bool DynamicAABBTree::overlap(const AABB &a, const AABB &b)
{
    if (a.max.x < b.min.x || b.max.x < a.min.x)
        return false;
    if (a.max.y < b.min.y || b.max.y < a.min.y)
        return false;
    if (a.max.z < b.min.z || b.max.z < a.min.z)
        return false;
    return true;
}

// Node pool
DynamicAABBTree::DynamicAABBTree()
{
    nodes_.reserve(16);
}

int DynamicAABBTree::allocateNode()
{
    if (freeList_ != NullNode)
    {
        int id = freeList_;
        freeList_ = nodes_[id].parent; // free list reuses the parent field as the linked list
        nodes_[id].parent = NullNode;
        nodes_[id].left = NullNode;
        nodes_[id].right = NullNode;
        nodes_[id].height = 0;
        nodes_[id].userData = -1;
        --freeCount_;
        return id;
    }
    Node n;
    nodes_.push_back(n);
    int id = static_cast<int>(nodes_.size()) - 1;
    return id;
}

void DynamicAABBTree::freeNode(int id)
{
    nodes_[id].height = -1;
    nodes_[id].parent = freeList_;
    freeList_ = id;
    ++freeCount_;
}

void DynamicAABBTree::clear()
{
    nodes_.clear();
    root_ = NullNode;
    freeList_ = NullNode;
    freeCount_ = 0;
}

// Leaf insert / remove
int DynamicAABBTree::insert(const AABB &aabb, int userData)
{
    int id = allocateNode();
    nodes_[id].aabb = aabb;
    nodes_[id].userData = userData;
    nodes_[id].height = 0;
    nodes_[id].left = NullNode;
    nodes_[id].right = NullNode;
    insertLeaf(id);
    return id;
}

void DynamicAABBTree::remove(int proxyId)
{
    if (proxyId < 0 || proxyId >= static_cast<int>(nodes_.size()))
        return;
    if (!nodes_[proxyId].isLeaf())
        return;
    removeLeaf(proxyId);
    freeNode(proxyId);
}

void DynamicAABBTree::update(int proxyId, const AABB &aabb)
{
    if (proxyId < 0 || proxyId >= static_cast<int>(nodes_.size()))
        return;
    if (!nodes_[proxyId].isLeaf())
        return;
    // Simple policy: remove + reinsert. Keeps userData, resets aabb.
    int userData = nodes_[proxyId].userData;
    removeLeaf(proxyId);
    nodes_[proxyId].aabb = aabb;
    nodes_[proxyId].userData = userData;
    nodes_[proxyId].height = 0;
    nodes_[proxyId].left = NullNode;
    nodes_[proxyId].right = NullNode;
    insertLeaf(proxyId);
}

// insertLeaf / removeLeaf — tree structure operations
void DynamicAABBTree::insertLeaf(int leaf)
{
    if (root_ == NullNode)
    {
        root_ = leaf;
        nodes_[leaf].parent = NullNode;
        return;
    }

    // --- 1. Descend from the root, choosing the sibling subtree that minimizes merged SA ---
    const AABB &leafAABB = nodes_[leaf].aabb;
    int index = root_;
    while (!nodes_[index].isLeaf())
    {
        int left = nodes_[index].left;
        int right = nodes_[index].right;

        float area = surfaceArea(nodes_[index].aabb);

        AABB combined = unionAABB(nodes_[index].aabb, leafAABB);
        float combinedArea = surfaceArea(combined);

        // Cost of creating a new parent = 2 * combinedArea
        float cost = 2.0f * combinedArea;

        // Cost of descending = 2 * (combinedArea - area)
        float inheritanceCost = 2.0f * (combinedArea - area);

        // Left subtree cost
        float costLeft;
        {
            AABB merged = unionAABB(leafAABB, nodes_[left].aabb);
            float newArea = surfaceArea(merged);
            if (nodes_[left].isLeaf())
                costLeft = newArea + inheritanceCost;
            else
                costLeft = (newArea - surfaceArea(nodes_[left].aabb)) + inheritanceCost;
        }
        // Right subtree cost
        float costRight;
        {
            AABB merged = unionAABB(leafAABB, nodes_[right].aabb);
            float newArea = surfaceArea(merged);
            if (nodes_[right].isLeaf())
                costRight = newArea + inheritanceCost;
            else
                costRight = (newArea - surfaceArea(nodes_[right].aabb)) + inheritanceCost;
        }

        if (cost < costLeft && cost < costRight)
            break;
        index = (costLeft < costRight) ? left : right;
    }

    int sibling = index;

    // --- 2. Create a new parent and hang leaf and sibling under it ---
    int oldParent = nodes_[sibling].parent;
    int newParent = allocateNode();
    nodes_[newParent].parent = oldParent;
    nodes_[newParent].userData = -1;
    nodes_[newParent].aabb = unionAABB(leafAABB, nodes_[sibling].aabb);
    nodes_[newParent].height = nodes_[sibling].height + 1;

    if (oldParent != NullNode)
    {
        if (nodes_[oldParent].left == sibling)
            nodes_[oldParent].left = newParent;
        else
            nodes_[oldParent].right = newParent;
    }
    else
    {
        root_ = newParent;
    }

    nodes_[newParent].left = sibling;
    nodes_[newParent].right = leaf;
    nodes_[sibling].parent = newParent;
    nodes_[leaf].parent = newParent;

    // --- 3. Fix AABB + height bottom-up, trying balance at each ancestor ---
    int walk = nodes_[leaf].parent;
    while (walk != NullNode)
    {
        walk = balance(walk);
        int l = nodes_[walk].left;
        int r = nodes_[walk].right;
        nodes_[walk].height = 1 + std::max(nodes_[l].height, nodes_[r].height);
        nodes_[walk].aabb = unionAABB(nodes_[l].aabb, nodes_[r].aabb);
        walk = nodes_[walk].parent;
    }
}

void DynamicAABBTree::removeLeaf(int leaf)
{
    if (leaf == root_)
    {
        root_ = NullNode;
        return;
    }

    int parent = nodes_[leaf].parent;
    int grand = nodes_[parent].parent;
    int sibling = (nodes_[parent].left == leaf) ? nodes_[parent].right : nodes_[parent].left;

    if (grand != NullNode)
    {
        if (nodes_[grand].left == parent)
            nodes_[grand].left = sibling;
        else
            nodes_[grand].right = sibling;
        nodes_[sibling].parent = grand;
        freeNode(parent);

        // Repair upward
        int walk = grand;
        while (walk != NullNode)
        {
            walk = balance(walk);
            int l = nodes_[walk].left;
            int r = nodes_[walk].right;
            nodes_[walk].aabb = unionAABB(nodes_[l].aabb, nodes_[r].aabb);
            nodes_[walk].height = 1 + std::max(nodes_[l].height, nodes_[r].height);
            walk = nodes_[walk].parent;
        }
    }
    else
    {
        root_ = sibling;
        nodes_[sibling].parent = NullNode;
        freeNode(parent);
    }
}

// ============================================================================
// balance — local left/right rotation keeping |heightL - heightR| <= 1.
// Classic AVL-style: if a subtree is unbalanced (diff >= 2), rotate the taller
// grandchild up. Follows Box2D b2DynamicTree::Balance, referenced but not ported.
// ============================================================================
int DynamicAABBTree::balance(int iA)
{
    Node &A = nodes_[iA];
    if (A.isLeaf() || A.height < 2)
        return iA;

    int iB = A.left;
    int iC = A.right;
    int balanceVal = nodes_[iC].height - nodes_[iB].height;

    // Right subtree too tall -> rotate left
    if (balanceVal > 1)
    {
        int iF = nodes_[iC].left;
        int iG = nodes_[iC].right;
        Node &C = nodes_[iC];
        Node &F = nodes_[iF];
        Node &G = nodes_[iG];

        // C takes A's position
        C.left = iA;
        C.parent = A.parent;
        A.parent = iC;

        if (C.parent != NullNode)
        {
            if (nodes_[C.parent].left == iA)
                nodes_[C.parent].left = iC;
            else
                nodes_[C.parent].right = iC;
        }
        else
        {
            root_ = iC;
        }

        // Rotate: reattach the shorter of F or G to A
        if (F.height > G.height)
        {
            C.right = iF;
            A.right = iG;
            G.parent = iA;
            A.aabb = unionAABB(nodes_[iB].aabb, G.aabb);
            C.aabb = unionAABB(A.aabb, F.aabb);
            A.height = 1 + std::max(nodes_[iB].height, G.height);
            C.height = 1 + std::max(A.height, F.height);
        }
        else
        {
            C.right = iG;
            A.right = iF;
            F.parent = iA;
            A.aabb = unionAABB(nodes_[iB].aabb, F.aabb);
            C.aabb = unionAABB(A.aabb, G.aabb);
            A.height = 1 + std::max(nodes_[iB].height, F.height);
            C.height = 1 + std::max(A.height, G.height);
        }
        return iC;
    }

    // Left subtree too tall -> rotate right
    if (balanceVal < -1)
    {
        int iD = nodes_[iB].left;
        int iE = nodes_[iB].right;
        Node &B = nodes_[iB];
        Node &D = nodes_[iD];
        Node &E = nodes_[iE];

        B.left = iA;
        B.parent = A.parent;
        A.parent = iB;

        if (B.parent != NullNode)
        {
            if (nodes_[B.parent].left == iA)
                nodes_[B.parent].left = iB;
            else
                nodes_[B.parent].right = iB;
        }
        else
        {
            root_ = iB;
        }

        if (D.height > E.height)
        {
            B.right = iD;
            A.left = iE;
            E.parent = iA;
            A.aabb = unionAABB(nodes_[iC].aabb, E.aabb);
            B.aabb = unionAABB(A.aabb, D.aabb);
            A.height = 1 + std::max(nodes_[iC].height, E.height);
            B.height = 1 + std::max(A.height, D.height);
        }
        else
        {
            B.right = iE;
            A.left = iD;
            D.parent = iA;
            A.aabb = unionAABB(nodes_[iC].aabb, D.aabb);
            B.aabb = unionAABB(A.aabb, E.aabb);
            A.height = 1 + std::max(nodes_[iC].height, D.height);
            B.height = 1 + std::max(A.height, E.height);
        }
        return iB;
    }

    return iA;
}

// Queries: AABB / ray / selfOverlap all use an iterative stack
void DynamicAABBTree::queryAABB(const AABB &aabb, const std::function<void(int)> &visitor) const
{
    if (root_ == NullNode)
        return;
    // Reuse the scratch stack to avoid per-call allocation; equivalent to a local vector.
    auto &stack = scratchQueryStack_;
    stack.clear();
    stack.push_back(root_);
    while (!stack.empty())
    {
        int id = stack.back();
        stack.pop_back();
        if (id == NullNode)
            continue;
        const Node &n = nodes_[id];
        if (!overlap(n.aabb, aabb))
            continue;
        if (n.isLeaf())
            visitor(n.userData);
        else
        {
            stack.push_back(n.left);
            stack.push_back(n.right);
        }
    }
}

// Slab test for ray vs AABB: intersect the [tmin_i, tmax_i] interval per world axis
static bool rayIntersectAABB(const AABB &aabb, const glm::vec3 &from,
                             const glm::vec3 &invDir, float maxT, float &tHit)
{
    float tmin = 0.0f;
    float tmax = maxT;
    for (int k = 0; k < 3; ++k)
    {
        float t1 = (aabb.min[k] - from[k]) * invDir[k];
        float t2 = (aabb.max[k] - from[k]) * invDir[k];
        if (t1 > t2)
            std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmin > tmax)
            return false;
    }
    tHit = tmin;
    return true;
}

void DynamicAABBTree::queryRay(const glm::vec3 &from, const glm::vec3 &dir, float maxT,
                               const std::function<bool(int)> &visitor) const
{
    if (root_ == NullNode)
        return;

    // Zero dir components use +inf to avoid division by zero; the slab test handles this degenerate case
    glm::vec3 invDir;
    for (int k = 0; k < 3; ++k)
    {
        float d = dir[k];
        invDir[k] = (std::abs(d) > 1e-30f) ? (1.0f / d) : std::numeric_limits<float>::infinity();
    }

    // Reuse the scratch stack to avoid per-call allocation.
    auto &stack = scratchQueryStack_;
    stack.clear();
    stack.push_back(root_);
    while (!stack.empty())
    {
        int id = stack.back();
        stack.pop_back();
        if (id == NullNode)
            continue;
        const Node &n = nodes_[id];
        float tHit;
        if (!rayIntersectAABB(n.aabb, from, invDir, maxT, tHit))
            continue;
        if (n.isLeaf())
        {
            if (!visitor(n.userData))
                return;
        }
        else
        {
            stack.push_back(n.left);
            stack.push_back(n.right);
        }
    }
}

void DynamicAABBTree::queryFrustum(const QueryPlane planes[6],
                                   const std::function<void(int)> &visitor) const
{
    // Plane-pruned traversal. For each node, conservatively cull the subtree
    // AABB against all 6 planes: take the p-vertex (corner chosen by the sign of
    // n's components); if its signed inside distance n*p + d < 0, the whole
    // subtree can be discarded. Descend only if all 6 planes pass; leaves invoke
    // visitor. Culling is conservative (may report false "visible"); the render
    // side re-runs a leaf-level testAABB, so results equal brute force.
    if (root_ == NullNode)
        return;

    auto &stack = scratchQueryStack_;
    stack.clear();
    stack.push_back(root_);
    while (!stack.empty())
    {
        int id = stack.back();
        stack.pop_back();
        if (id == NullNode)
            continue;
        const Node &n = nodes_[id];
        bool outside = false;
        for (int i = 0; i < 6; ++i)
        {
            const glm::vec3 &pn = planes[i].n;
            // p-vertex: corner most toward the plane's positive side
            glm::vec3 p(
                pn.x >= 0.0f ? n.aabb.max.x : n.aabb.min.x,
                pn.y >= 0.0f ? n.aabb.max.y : n.aabb.min.y,
                pn.z >= 0.0f ? n.aabb.max.z : n.aabb.min.z);
            if (glm::dot(pn, p) + planes[i].d < 0.0f)
            {
                outside = true;
                break;
            }
        }
        if (outside)
            continue;
        if (n.isLeaf())
            visitor(n.userData);
        else
        {
            stack.push_back(n.left);
            stack.push_back(n.right);
        }
    }
}

void DynamicAABBTree::selfOverlap(const std::function<void(int, int)> &visitor) const
{
    if (root_ == NullNode || nodes_[root_].isLeaf())
        return;

    // Stack holds node pairs (a, b); either may be a leaf or an internal node.
    // For each internal node v consider the pair (v.left, v.right) plus recursive
    // self-collision inside v.left / v.right. Reuses the scratch pair stack;
    // PairFrame is declared in the header.
    auto &stack = scratchPairStack_;
    stack.clear();

    // Without self-descent from the root, collisions inside same-side subtrees
    // would be missed: for each internal node n, push (n.left, n.right) and
    // recurse into n.left / n.right for their internal self-overlap.
    auto &singleStack = scratchSingleStack_;
    singleStack.clear();
    singleStack.push_back(root_);
    while (!singleStack.empty())
    {
        int id = singleStack.back();
        singleStack.pop_back();
        const Node &n = nodes_[id];
        if (n.isLeaf())
            continue;
        stack.push_back({n.left, n.right});
        singleStack.push_back(n.left);
        singleStack.push_back(n.right);
    }

    while (!stack.empty())
    {
        PairFrame f = stack.back();
        stack.pop_back();
        const Node &A = nodes_[f.a];
        const Node &B = nodes_[f.b];
        if (!overlap(A.aabb, B.aabb))
            continue;
        if (A.isLeaf() && B.isLeaf())
        {
            visitor(A.userData, B.userData);
        }
        else if (A.isLeaf())
        {
            stack.push_back({f.a, B.left});
            stack.push_back({f.a, B.right});
        }
        else if (B.isLeaf())
        {
            stack.push_back({A.left, f.b});
            stack.push_back({A.right, f.b});
        }
        else
        {
            // Expand the four A x B pairs
            stack.push_back({A.left, B.left});
            stack.push_back({A.left, B.right});
            stack.push_back({A.right, B.left});
            stack.push_back({A.right, B.right});
        }
    }
}
