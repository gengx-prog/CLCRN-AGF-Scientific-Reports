import torch
import torch.nn as nn


class MultiScaleGraphPropagation(nn.Module):
    """Training-free multi-scale propagation inspired by SGP."""

    def __init__(self, sparse_idx, geodesic, node_num, orders=2):
        super().__init__()
        self.node_num = node_num
        self.orders = max(int(orders), 0)
        propagation = self._build_normalized_operator(sparse_idx, geodesic, node_num)
        self.register_buffer("propagation", propagation)

    @staticmethod
    def _build_normalized_operator(sparse_idx, geodesic, node_num):
        indices = sparse_idx.clone().long()
        values = torch.exp(-geodesic.float())

        loop_index = torch.arange(node_num, device=indices.device, dtype=torch.long)
        loop_indices = torch.stack([loop_index, loop_index], dim=0)
        loop_values = torch.ones(node_num, device=values.device, dtype=values.dtype)

        indices = torch.cat([indices, loop_indices], dim=1)
        values = torch.cat([values, loop_values], dim=0)

        operator = torch.sparse_coo_tensor(indices, values, (node_num, node_num)).coalesce()
        row_sum = torch.sparse.sum(operator, dim=1).to_dense().clamp(min=1e-6)
        normalized_values = operator.values() / row_sum[operator.indices()[0]]

        return torch.sparse_coo_tensor(
            operator.indices(),
            normalized_values,
            operator.size(),
        ).coalesce()

    def _propagate_once(self, x):
        batch_size, seq_len, node_num, feat_dim = x.shape
        flat = x.permute(2, 0, 1, 3).reshape(node_num, -1)
        propagated = torch.sparse.mm(self.propagation, flat)
        propagated = propagated.reshape(node_num, batch_size, seq_len, feat_dim)
        return propagated.permute(1, 2, 0, 3)

    def forward(self, x):
        scales = [x]
        current = x
        for _ in range(self.orders):
            current = self._propagate_once(current)
            scales.append(current)
        return torch.cat(scales, dim=-1)
