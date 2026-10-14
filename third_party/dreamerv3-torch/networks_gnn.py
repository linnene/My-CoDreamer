"""CoDreamer 的图神经网络层 —— agent 之间的"通信"就在这里发生。

================================================================================
归属说明
================================================================================
本文件**不属于上游 dreamerv3-torch**（不在 UPSTREAM_MANIFEST.sha256 里），
是 CoDreamer 复现的一部分，由我们编写。

================================================================================
通信到底是什么
================================================================================
论文里没有"消息向量"这种独立机制。所谓通信就是：

    每个 agent 把自己的隐状态当作图上的一个节点特征，
    然后沿边做一次**邻域聚合**，聚合结果进自己的更新。

    论文（§2.2.2）："All elements from IDreamer remain unchanged except for the
    addition of k GNN layers within the RSSM and prediction heads to facilitate
    communication."

所以"通信"= **GNN 层**。

================================================================================
为什么要分成"同时"的一层
================================================================================
一个容易被误解的点：GNN 不是"agent 1 先把自己的状态告诉 agent 2，然后 2 再用"。
那样会有先后顺序，先手吃亏。真实的计算是**同时**的：

    new_1 = f( old_1 , { old_j : j ∈ N_1 } )
    new_2 = f( old_2 , { old_j : j ∈ N_2 } )
             ↑         ↑
          各自旧值   邻居的旧值

两边都只读**更新前**的值，一次算完。k 层 = 在同一时刻内传 k 跳。
所以这里没有任何循环、没有先后 —— 与"时间轴上的 RNN"是**正交的两个轴**。

================================================================================
算子：GATv2 + 边特征
================================================================================
论文（§2.2.1）：
    "All graphs have edge features representing the relative distances between
     agents, which are used with observations in the **GAT V2** to calculate
     attention coefficients for neighborhood aggregation."

所以注意力系数必须**同时**依赖两端的节点特征和边特征，这正是 GATv2 相对
GAT 的改动（GAT 把 a^T[Wh_i || Wh_j] 拆成两项，导致注意力对邻居特征单调，
表达力受限）：

    GATv2:  e_ij = a^T · LeakyReLU( W_l h_i + W_r h_j + W_e E_ij )
    GAT:    e_ij = LeakyReLU( a_l^T W h_i + a_r^T W h_j )

本实现用 GATv2 形式。

================================================================================
防过平滑（论文明确要求，容易漏）
================================================================================
    "Each GNN layer performs node updates and neighborhood aggregation,
     **concatenating the original node features to the output** to help the world
     model distinguish agents and **prevent over-smoothing**"

不加这个跳连，k 层之后所有节点会趋同，模型就分不出谁是谁了。所以每层的输出是

    out_i = act( W_out · concat( aggregated_i , h_i ) )

输入特征被拼在输出里，身份信息得以保留。输入输出同维，层可以任意叠。

================================================================================
孤立 agent
================================================================================
论文承认邻接矩阵里会出现没有邻居的 agent（Estimate Game 里 rho=0.6，必然会
出现），并说：

    "Depending on the adjacency matrix A, agents can be processed independently,
     **effectively transforming CoDreamer into IDreamer for those agents**."

本实现给每个节点加自环（标准 GATv2 做法）。孤立 agent 的 softmax 只剩自己那一项，
聚合退化为对自身的变换，不会出现 0/0。
"""

from __future__ import annotations

import torch
from torch import nn

import tools

__all__ = ["GATv2Layer", "GraphAggregator"]


class GATv2Layer(nn.Module):
    """一层 GATv2 + 原特征跳连。

    张量布局（agent 轴是**倒数第二个**轴，与数据管线的 (B,T,N,·) 一致）::

        node      (..., N, F_in)      节点特征
        adjacency (..., N, N)         邻接矩阵，0/1，无自环（自环由本层内部加）
        edge      (..., N, N, F_edge) 边特征（可为 None）

    返回 (..., N, F_out)。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 0,
        heads: int = 1,
        act: str = "SiLU",
        norm: bool = True,
        negative_slope: float = 0.2,
        self_loops: bool = True,
    ) -> None:
        super().__init__()
        if in_dim % heads or out_dim % heads:
            raise ValueError(f"in/out dim must be divisible by heads: {in_dim} {out_dim} {heads}")
        self._in = in_dim
        self._out = out_dim
        self._heads = heads
        self._head_dim = out_dim // heads
        self._edge_dim = edge_dim
        self._self_loops = self_loops
        self._negative_slope = negative_slope

        # GATv2 的三项：左右节点各一个线性，边特征单独一个线性
        self._lin_l = nn.Linear(in_dim, out_dim, bias=False)
        self._lin_r = nn.Linear(in_dim, out_dim, bias=False)
        self._lin_v = nn.Linear(in_dim, out_dim, bias=False)
        self._lin_e = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None
        # 注意力向量 a：每个 head 一个，作用在 LeakyReLU 之后的 out_dim 上
        self._att = nn.Parameter(torch.zeros(heads, self._head_dim))

        # 原特征跳连：把输入拼在输出上再投影，防止 over-smoothing（论文要求）
        self._skip = nn.Linear(in_dim + out_dim, out_dim, bias=False)
        self._norm = nn.LayerNorm(out_dim, eps=1e-03) if norm else nn.Identity()
        self._act = getattr(torch.nn, act)()
        self.apply(tools.weight_init)
        # 注意力向量必须**非零**初始化。若全零，则所有 logits = 0，softmax 退化成
        # 对邻居做均匀平均 —— GATv2 的核心机制（注意力依赖节点与边特征）在初始化时
        # 完全失效，边特征也拿不到任何梯度。（曾被自检的"边特征影响输出"一条抓到。）
        # 取小标准差正态：既有非零梯度，又不会一上来就让 softmax 饱和。
        nn.init.normal_(self._att, mean=0.0, std=0.1)

    def forward(self, node, adjacency, edge=None):
        *lead, n, _ = node.shape
        a = adjacency
        if self._self_loops:
            eye = torch.eye(n, device=a.device, dtype=a.dtype)
            a = torch.clamp(a + eye, max=1.0)
        # (..., N, N) -> (..., N, N, 1) 以便广播
        mask = a.unsqueeze(-1)

        # --- GATv2 打分: e_ij = a^T LeakyReLU(W_l h_i + W_r h_j + W_e E_ij)
        left = self._lin_l(node)                                   # (..., N, H*D)
        right = self._lin_r(node)                                  # (..., N, H*D)
        score = left.unsqueeze(-2) + right.unsqueeze(-3)           # (..., N, N, H*D)
        if self._lin_e is not None:
            if edge is None:
                raise ValueError("edge features required but edge=None")
            score = score + self._lin_e(edge.to(score.dtype))
        score = torch.nn.functional.leaky_relu(score, self._negative_slope)
        score = score.reshape(*score.shape[:-1], self._heads, self._head_dim)
        score = (score * self._att).sum(-1)                        # (..., N, N, heads)

        # --- 只保留真实存在的边；孤立节点靠自环保住一项，不会 0/0
        score = score.masked_fill(mask < 0.5, float("-inf"))
        alpha = torch.softmax(score, dim=-2)                       # 沿邻居维归一
        # 数值保护：若某行全被 mask（理论上自环后不会），softmax 会出 NaN
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # --- 聚合: y_i = sum_j alpha_ij * W_v h_j
        value = self._lin_v(node).reshape(*node.shape[:-1], self._heads, self._head_dim)
        # (..., N, N, H, D) = (..., N, 1, H, D) * (..., 1, N, H, 1) 的和 -> 交换求和顺序
        agg = torch.einsum("...ijh,...jhd->...ihd", alpha, value)
        agg = agg.reshape(*node.shape[:-1], self._out)

        # --- 原特征跳连（论文的防过平滑），并保证形状可叠层
        out = torch.cat([agg, node], dim=-1)
        out = self._act(self._norm(self._skip(out)))
        return out


class GraphAggregator(nn.Module):
    """k 层 GATv2 堆叠。`layers == 0` 时是**恒等变换**。

    `layers == 0` 必须是严格的恒等，这是 IDreamer 与 CoDreamer 之间唯一开关的
    前提：关掉它，整条链路必须退化成 IDreamer，一个数都不能差。
    """

    def __init__(
        self,
        dim: int,
        layers: int = 0,
        edge_dim: int = 0,
        heads: int = 1,
        act: str = "SiLU",
        norm: bool = True,
    ) -> None:
        super().__init__()
        self._layers = int(layers)
        self._edge_dim = int(edge_dim)
        self.layers = nn.ModuleList(
            [
                GATv2Layer(dim, dim, edge_dim=edge_dim, heads=heads, act=act, norm=norm)
                for _ in range(self._layers)
            ]
        )

    @property
    def num_layers(self) -> int:
        return self._layers

    def forward(self, node, adjacency, edge=None):
        if self._layers == 0:
            # 恒等：连 dtype/device 都不动，保证"关掉 GNN == IDreamer"
            return node
        out = node
        for layer in self.layers:
            out = layer(out, adjacency, edge)
        return out


# ============================================================================
# 自检 —— 本地不跑实验，断言写死在代码里。
#   在 third_party/dreamerv3-torch 目录下： python -m networks_gnn
# ============================================================================


def _self_test() -> None:
    import numpy as np

    failures: list[str] = []

    def check(name, cond, detail=""):
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("networks_gnn 自检")
    print("-" * 74)
    torch.manual_seed(0)
    B, T, N, F, Fe = 2, 3, 4, 8, 1

    node = torch.randn(B, T, N, F)
    # 一个确定的邻接矩阵：0-1、1-2 相连；3 是孤岛
    A = torch.zeros(N, N)
    A[0, 1] = A[1, 0] = 1
    A[1, 2] = A[2, 1] = 1
    A = A.expand(B, T, N, N).contiguous()
    E = (torch.rand(B, T, N, N, Fe) * (A > 0).unsqueeze(-1)).contiguous()

    # --- 0 层必须是严格恒等
    agg0 = GraphAggregator(F, layers=0, edge_dim=Fe)
    out0 = agg0(node, A, E)
    check("layers=0 是同一对象（真恒等）", out0 is node)

    # --- 形状
    for layers in (1, 2, 3):
        agg = GraphAggregator(F, layers=layers, edge_dim=Fe)
        out = agg(node, A, E)
        check(f"layers={layers} 形状不变", out.shape == (B, T, N, F), str(out.shape))
        check(f"layers={layers} 无 NaN", bool(torch.isfinite(out).all()))

    # --- 同时性：改变 agent 2 的输入应当影响 1（相邻），但**不应**影响 3（孤岛）
    agg = GraphAggregator(F, layers=1, edge_dim=Fe)
    out_a = agg(node, A, E)
    node_b = node.clone()
    node_b[:, :, 2, :] += 5.0
    out_b = agg(node_b, A, E)
    touch1 = (out_a[:, :, 1] - out_b[:, :, 1]).abs().max().item()
    touch3 = (out_a[:, :, 3] - out_b[:, :, 3]).abs().max().item()
    check("邻居变化会传到相邻 agent", touch1 > 1e-6, f"Δ={touch1:.3e}")
    check("孤岛 agent 完全不受影响", touch3 == 0.0, f"Δ={touch3:.3e}")

    # --- 孤立节点的输出必须是有限的（softmax 不能 0/0）
    A_iso = torch.zeros(B, T, N, N)
    out_iso = agg(node, A_iso, torch.zeros(B, T, N, N, Fe))
    check("全孤岛时输出有限", bool(torch.isfinite(out_iso).all()))

    # --- 2 层应当把信息传到 2 跳：改 agent 0 应当能影响 agent 2
    agg2 = GraphAggregator(F, layers=2, edge_dim=Fe)
    o2a = agg2(node, A, E)
    node_c = node.clone()
    node_c[:, :, 0, :] += 5.0
    o2b = agg2(node_c, A, E)
    d2 = (o2a[:, :, 2] - o2b[:, :, 2]).abs().max().item()
    check("2 层能传 2 跳（0 影响 2）", d2 > 1e-6, f"Δ={d2:.3e}")

    # 1 层时 0 对 2 的影响只经由 1，仍然存在（消息传 1 跳后 2 看到的是旧的 1），
    # 所以这里只验证"层数越多、远端受影响越强"，不验证 1 层为 0
    agg1 = GraphAggregator(F, layers=1, edge_dim=Fe)
    o1a = agg1(node, A, E)
    o1b = agg1(node_c, A, E)
    d1 = (o1a[:, :, 2] - o1b[:, :, 2]).abs().max().item()
    check("层数增加会放大远端影响", d2 > d1, f"1层 Δ={d1:.3e}  2层 Δ={d2:.3e}")

    # --- 防过平滑：不同节点的输出不应趋于相同
    agg3 = GraphAggregator(F, layers=3, edge_dim=Fe)
    o3 = agg3(node, A, E)
    spread = o3.std(dim=2).mean().item()
    check("3 层后节点间仍有区分度（防过平滑）", spread > 1e-3, f"std={spread:.3e}")

    # --- 边特征确实起作用（若注意力向量初始化为 0，这条会挂）
    agg_e = GraphAggregator(F, layers=1, edge_dim=Fe)
    o_e1 = agg_e(node, A, E)
    o_e2 = agg_e(node, A, torch.ones_like(E))
    d_e = (o_e1 - o_e2).abs().max().item()
    check("边特征影响输出", d_e > 1e-6, f"Δ={d_e:.3e}")
    # 并且注意力分布不是均匀的（均匀 = GATv2 机制失效）
    alpha_ok = True
    for m in agg_e.modules():
        if isinstance(m, GATv2Layer):
            alpha_ok = alpha_ok and float(m._att.detach().abs().max()) > 0
    check("注意力向量非零（否则退化为均匀平均）", alpha_ok)

    # --- 无自环选项
    layer_no_loop = GATv2Layer(F, F, edge_dim=Fe, self_loops=False)
    try:
        o_nl = layer_no_loop(node, A, E)
        check("self_loops=False 正常返回", bool(torch.isfinite(o_nl).all()))
    except Exception as e:  # noqa: BLE001
        check("self_loops=False 正常返回", False, repr(e))

    # --- 梯度能回流
    agg_g = GraphAggregator(F, layers=2, edge_dim=Fe)
    node_g = node.clone().requires_grad_(True)
    agg_g(node_g, A, E).sum().backward()
    check("梯度回流到输入", node_g.grad is not None and bool(torch.isfinite(node_g.grad).all()))
    has_grad = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in agg_g.parameters() if p.requires_grad
    )
    check("所有参数都拿到梯度", has_grad)

    # --- 参数量级：应当在 0.5M 附近（论文 actor/critic 比我们各多约 0.5M）
    big = GraphAggregator(512, layers=1, edge_dim=1)
    n_par = sum(p.numel() for p in big.parameters())
    print(f"       参考：1 层 GATv2(dim=512) 参数量 = {n_par:,}")
    check("1 层 GATv2 参数量在合理量级", 0.5e6 < n_par < 1.5e6, f"{n_par:,}")

    print("-" * 74)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        raise SystemExit(1)
    print("全部通过。")


if __name__ == "__main__":
    _self_test()
