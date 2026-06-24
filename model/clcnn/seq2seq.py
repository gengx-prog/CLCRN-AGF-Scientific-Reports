import torch

class Seq2SeqAttrs:
    def __init__(self, sparse_idx, angle_ratio, geodesic, **model_kwargs):
        self.sparse_idx = sparse_idx
        self.max_view = int(model_kwargs.get('max_view', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.node_num = int(model_kwargs.get('node_num', 6))
        self.layer_num = int(model_kwargs.get('layer_num', 2))
        self.rnn_units = int(model_kwargs.get('rnn_units', 32))
        self.input_dim = int(model_kwargs.get('input_dim', 2))
        self.output_dim = int(model_kwargs.get('output_dim', 2))
        self.seq_len = int(model_kwargs.get('seq_len', 12))
        self.lck_structure = model_kwargs.get('lck_structure', model_kwargs.get('lckstructure', [4,8]))
        self.embed_dim = int(model_kwargs.get('embed_dim', 16))
        self.context_dim = int(model_kwargs.get('context_dim', 0))
        self.use_context_embedding = bool(model_kwargs.get('use_context_embedding', self.context_dim > 0))
        self.use_context_in_signal_branch = bool(model_kwargs.get('use_context_in_signal_branch', self.context_dim > 0))
        self.context_embed_dim = int(model_kwargs.get('context_embed_dim', self.embed_dim if self.context_dim > 0 else 0))
        self.use_periodic_time_encoding = bool(model_kwargs.get('use_periodic_time_encoding', False))
        self.periodic_time_hidden_dim = int(model_kwargs.get('periodic_time_hidden_dim', max(self.embed_dim, 8)))
        self.periodic_time_use_year = bool(model_kwargs.get('periodic_time_use_year', True))
        self.periodic_time_use_geo = bool(model_kwargs.get('periodic_time_use_geo', False))
        self.periodic_time_init_scale = float(model_kwargs.get('periodic_time_init_scale', 0.25))
        self.use_sgp_encoder = bool(model_kwargs.get('use_sgp_encoder', False))
        self.sgp_orders = int(model_kwargs.get('sgp_orders', 0))
        self.use_asttn_encoder = bool(model_kwargs.get('use_asttn_encoder', False))
        self.asttn_node_dim = int(model_kwargs.get('asttn_node_dim', 10))
        self.asttn_hidden_dim = int(model_kwargs.get('asttn_hidden_dim', self.embed_dim))
        self.asttn_heads = int(model_kwargs.get('asttn_heads', 2))
        self.asttn_topk = int(model_kwargs.get('asttn_topk', 8))
        self.asttn_fusion_mode = model_kwargs.get('asttn_fusion_mode', 'gated')
        self.location_dim = int(model_kwargs.get('location_dim', 16))
        self.horizon = int(model_kwargs.get('horizon', 16))
        self.hidden_units = int(model_kwargs.get('hidden_units', 16))
        self.block_num = int(model_kwargs.get('block_num', 2))
        signal_branch_context_dim = self.context_dim if self.use_context_in_signal_branch else 0
        base_signal_dim = self.input_dim + signal_branch_context_dim
        propagated_signal_dim = base_signal_dim * (self.sgp_orders + 1) if self.use_sgp_encoder else base_signal_dim
        context_embedding_dim = self.context_embed_dim if (self.context_dim > 0 and self.use_context_embedding) else 0
        adaptive_attention_dim = self.asttn_hidden_dim if self.use_asttn_encoder else 0
        self.encoder_input_dim = int(
            model_kwargs.get(
                'encoder_input_dim',
                self.embed_dim + self.embed_dim + context_embedding_dim + propagated_signal_dim + adaptive_attention_dim,
            )
        )
        angle_ratio = torch.sparse.FloatTensor(
            self.sparse_idx, 
            angle_ratio, 
            (self.node_num,self.node_num)
            ).to_dense() 
        self.angle_ratio = angle_ratio + torch.eye(*angle_ratio.shape).to(angle_ratio.device)
        self.geodesic =  torch.sparse.FloatTensor(
            self.sparse_idx, 
            geodesic, 
            (self.node_num,self.node_num)
            ).to_dense()
