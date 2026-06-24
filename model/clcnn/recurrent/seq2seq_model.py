import numpy as np
import torch
import torch.nn as nn
from ..adaptive_attention import AdaptiveGraphSpatialAttention
from ..clconv import CLConv
from ..seq2seq import Seq2SeqAttrs
from ..sgp import MultiScaleGraphPropagation
from ..temporal_encoding import PeriodicTimeEncoder
from .encoder import EncoderModel
from .decoder import DecoderModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CLCRNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, loc_info, sparse_idx, geodesic, angle_ratio, logger=None, **model_kwargs):
        '''
        Conditional Local Convolution Recurrent Network, implemented based on DCRNN,
        Args:
            loc_info (torch.Tensor): location infomation of each nodes, with the shape (node_num, location_dim). For sphercial signals, location_dim=2.
            sparse_idx (torch.Tensor): sparse_idx with the shape (2, node_num * nbhd_num).
            geodesic (torch.Tensor): geodesic distance between each point and its neighbors, with the shape (node_num * nbhd_num), corresponding to sparse_idx.
            angle_ratio (torch.Tensor): the defined angle ratio contributing to orientation density, with the shape (node_num * nbhd_num), corresponding to sparse_idx.
            model_kwargs (dict): Other model args see the config.yaml.
        '''
        super().__init__()
        Seq2SeqAttrs.__init__(self, sparse_idx, angle_ratio, geodesic, **model_kwargs)
        self.node_embeddings = nn.Parameter(torch.randn(self.node_num, self.embed_dim))
        self.feature_embedding = nn.Linear(self.input_dim, self.embed_dim)
        self.context_embedding = None
        if self.context_dim > 0 and self.use_context_embedding:
            self.context_embedding = nn.Linear(self.context_dim, self.context_embed_dim)
        self.periodic_time_encoder = None
        self.periodic_time_scale = None
        if self.use_periodic_time_encoding:
            if self.context_dim < 4:
                raise ValueError("Periodic time encoding requires context_dim >= 4 for year/month/day/hour inputs.")
            self.periodic_time_encoder = PeriodicTimeEncoder(
                output_dim=self.embed_dim,
                hidden_dim=self.periodic_time_hidden_dim,
                use_year_trend=self.periodic_time_use_year,
                include_geo=self.periodic_time_use_geo,
            )
            self.periodic_time_scale = nn.Parameter(torch.tensor(self.periodic_time_init_scale, dtype=torch.float32))
        self.sgp_encoder = None
        if self.use_sgp_encoder:
            self.sgp_encoder = MultiScaleGraphPropagation(
                sparse_idx=sparse_idx,
                geodesic=geodesic,
                node_num=self.node_num,
                orders=self.sgp_orders,
            )
        self.asttn_encoder = None
        if self.use_asttn_encoder:
            self.asttn_encoder = AdaptiveGraphSpatialAttention(
                node_num=self.node_num,
                input_dim=self.embed_dim * 2,
                output_dim=self.asttn_hidden_dim,
                adaptive_node_dim=self.asttn_node_dim,
                num_heads=self.asttn_heads,
                topk=self.asttn_topk,
                fusion_mode=self.asttn_fusion_mode,
            )

        self.conv_ker = CLConv(
            self.location_dim, 
            self.sparse_idx,
            self.node_num,
            self.lck_structure, 
            loc_info, 
            self.angle_ratio, 
            self.geodesic,
            self.max_view
        )
        self.encoder_model = EncoderModel(sparse_idx, angle_ratio, geodesic, self.conv_ker, **model_kwargs)
        self.decoder_model = DecoderModel(sparse_idx, angle_ratio, geodesic, self.conv_ker, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def get_kernel(self):
        return self.conv_ker

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):

        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.node_num, self.output_dim))
        go_symbol = go_symbol.to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                    decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def embedding(self, inputs):
        # inputs: b, l, n, f
        # outputs: b, l, n, encoder_input_dim
        batch_size, seq_len, node_num, feature_size = inputs.shape
        expected_feature_size = self.input_dim + self.context_dim
        if feature_size < expected_feature_size:
            raise ValueError(
                f"Expected at least {expected_feature_size} input features, got {feature_size}."
            )

        signal_inputs = inputs[..., :self.input_dim]
        context_inputs = None
        if self.context_dim > 0:
            context_inputs = inputs[..., self.input_dim:self.input_dim + self.context_dim]

        feature_emb = self.feature_embedding(signal_inputs)
        node_emb = self.node_embeddings[None, None, :, :].expand(batch_size, seq_len, node_num, self.embed_dim)
        if self.periodic_time_encoder is not None and context_inputs is not None:
            feature_emb = feature_emb + self.periodic_time_scale * self.periodic_time_encoder(context_inputs)
        adaptive_spatial_emb = None
        if self.asttn_encoder is not None:
            adaptive_inputs = torch.cat([feature_emb, node_emb], dim=-1)
            adaptive_spatial_emb = self.asttn_encoder(adaptive_inputs)

        if context_inputs is not None and self.use_context_in_signal_branch:
            base_signal = torch.cat([signal_inputs, context_inputs], dim=-1)
        else:
            base_signal = signal_inputs
        propagated_signal = self.sgp_encoder(base_signal) if self.sgp_encoder is not None else base_signal

        parts = [feature_emb, node_emb]
        if self.context_embedding is not None and context_inputs is not None:
            parts.append(self.context_embedding(context_inputs))
        parts.append(propagated_signal)
        if adaptive_spatial_emb is not None:
            parts.append(adaptive_spatial_emb)
        return torch.cat(parts, dim=-1)

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor, input_dim)
        :param labels: shape (horizon, batch_size, num_sensor, output_dim)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.node_num * self.output_dim)
        """
        embedding = self.embedding(inputs)
        encoder_hidden_state = self.encoder(embedding)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs
