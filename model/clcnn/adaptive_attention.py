import torch
import torch.nn as nn


class AdaptiveGraphSpatialAttention(nn.Module):
    """ASTTN-inspired adaptive graph fusion with gated residual mixing."""

    def __init__(
        self,
        node_num,
        input_dim,
        output_dim,
        adaptive_node_dim=10,
        num_heads=2,
        topk=8,
        fusion_mode="gated",
    ):
        super().__init__()
        self.node_num = node_num
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.topk = max(1, min(topk, node_num))
        self.fusion_mode = fusion_mode
        valid_modes = {"gated", "mean", "mixed_only", "residual_only"}
        if self.fusion_mode not in valid_modes:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}. Choose from {sorted(valid_modes)}.")

        self.node_emb_src = nn.Parameter(torch.empty(node_num, adaptive_node_dim))
        self.node_emb_dst = nn.Parameter(torch.empty(node_num, adaptive_node_dim))

        self.out_proj = nn.Linear(input_dim, output_dim)
        self.residual_proj = nn.Linear(input_dim, output_dim)
        self.gate_proj = nn.Linear(output_dim * 2, output_dim) if self.fusion_mode == "gated" else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb_src)
        nn.init.xavier_uniform_(self.node_emb_dst)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)
        if self.gate_proj is not None:
            nn.init.xavier_uniform_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)

    def _adaptive_topk(self):
        adjacency = torch.softmax(self.node_emb_src @ self.node_emb_dst.T, dim=-1)
        topk_without_self = max(0, self.topk - 1)
        self_indices = torch.arange(self.node_num, device=adjacency.device).unsqueeze(-1)
        if topk_without_self > 0:
            candidate_adjacency = adjacency.masked_fill(
                torch.eye(self.node_num, dtype=torch.bool, device=adjacency.device),
                -torch.inf,
            )
            topk_values, topk_indices = torch.topk(candidate_adjacency, k=topk_without_self, dim=-1)
            self_values = adjacency.gather(1, self_indices)
            topk_indices = torch.cat([self_indices, topk_indices], dim=-1)
            topk_values = torch.cat([self_values, topk_values], dim=-1)
        else:
            topk_indices = self_indices
            topk_values = adjacency.gather(1, self_indices)
        topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return topk_indices, topk_values

    def forward(self, x):
        # x: (batch, seq_len, node_num, input_dim)
        batch_size, seq_len, node_num, _ = x.shape
        if node_num != self.node_num:
            raise ValueError(f"Expected {self.node_num} nodes, got {node_num}.")

        topk_indices, topk_values = self._adaptive_topk()
        flat_neighbor_indices = topk_indices.reshape(-1)
        neighbor_x = x[:, :, flat_neighbor_indices, :].reshape(
            batch_size, seq_len, node_num, self.topk, x.size(-1)
        )
        mixed = torch.sum(neighbor_x * topk_values[None, None, :, :, None], dim=3)
        mixed = self.out_proj(mixed)

        residual = self.residual_proj(x)
        if self.fusion_mode == "gated":
            gate = torch.sigmoid(self.gate_proj(torch.cat([residual, mixed], dim=-1)))
            return gate * mixed + (1.0 - gate) * residual
        if self.fusion_mode == "mean":
            return 0.5 * (mixed + residual)
        if self.fusion_mode == "mixed_only":
            return mixed
        return residual
