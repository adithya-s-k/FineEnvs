import torch
from transformers import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2DecoderLayer
from packing import install


def test_packed_convolution_matches_independent_outputs_and_gradients():
    torch.manual_seed(7)
    config=Lfm2Config(hidden_size=32,block_dim=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,layer_types=['conv'],conv_L_cache=3)
    layer=Lfm2DecoderLayer(config,0).double()
    x=torch.randn(1,13,32,dtype=torch.float64,requires_grad=True)
    a=layer(x[:,:6],position_ids=torch.arange(6)[None])
    b=layer(x[:,6:],position_ids=torch.arange(7)[None])
    expected=torch.cat([a,b],dim=1)
    grad=torch.autograd.grad(expected.square().sum(),x)[0]
    install()
    actual=layer(x,position_ids=torch.cat([torch.arange(6),torch.arange(7)])[None])
    actual_grad=torch.autograd.grad(actual.square().sum(),x)[0]
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(actual_grad,grad,rtol=1e-10,atol=1e-10)
