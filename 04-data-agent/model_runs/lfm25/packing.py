"""Keep LFM convolution state independent across packed training sequences."""
import torch


def install():
    from transformers.models.lfm2.modeling_lfm2 import Lfm2DecoderLayer
    if getattr(Lfm2DecoderLayer, '_openenv_packing', False):
        return
    original = Lfm2DecoderLayer.forward
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        if self.is_attention_layer or position_ids is None or past_key_values is not None:
            return original(self, hidden_states, position_embeddings=position_embeddings,
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, **kwargs)
        if hidden_states.shape[0] != 1:
            raise ValueError('Packed LFM convolution adapter expects one flattened row')
        starts = (position_ids[0] == 0).nonzero(as_tuple=True)[0].tolist()
        if not starts or starts[0] != 0:
            raise ValueError('Packed sequence must begin at position zero')
        bounds = starts + [hidden_states.shape[1]]
        residual = hidden_states
        normed = self.operator_norm(hidden_states)
        pieces = []
        for start, end in zip(bounds, bounds[1:]):
            mask = attention_mask[:, start:end] if attention_mask is not None else None
            pieces.append(self.conv(hidden_states=normed[:, start:end],
                past_key_values=None, attention_mask=mask))
        hidden_states = residual + torch.cat(pieces, dim=1)
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
    Lfm2DecoderLayer.forward = forward
    Lfm2DecoderLayer._openenv_packing = True
