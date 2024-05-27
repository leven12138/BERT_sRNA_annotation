"""
实现 transformer 的 encode 部分
"""
from torch import nn
from module.base_transformer.layers import HalfFeedForward, AddNorm, AttentionAddNorm


class EncoderLayer(nn.Module):
    """
    注意这个结构和 bert 的结构完全一致, 在 bert 中被称作 BertLayer。
    """

    def __init__(self, 
                 hidden_size,                  # 隐藏层大小
                 hidden_act,                   # 隐藏层激活函数
                 intermediate_size,
                 num_attention_heads,          # 注意力头数量
                 attention_probs_dropout_prob, # attention prob 的 dropout 比例
                 hidden_dropout_prob,          # dropout 比例
                 layer_norm_eps,          
                ): 
        super(EncoderLayer, self).__init__()
        # Multi-Head Attention
        self.attention = AttentionAddNorm(hidden_size, num_attention_heads, 
                                          attention_probs_dropout_prob, 
                                          hidden_dropout_prob, layer_norm_eps)

        # Feed Forward + Add & Norm
        self.intermediate = HalfFeedForward(hidden_size, intermediate_size, hidden_act)
        self.output = AddNorm(intermediate_size, hidden_size,
                              hidden_dropout_prob, layer_norm_eps)

    def forward(self, query_tensor, key_tensor, value_tensor, attention_mask=None):
        # Multi-Head Attention
        attention_output = self.attention(query_tensor, key_tensor, value_tensor, attention_mask)

        # Feed Forward + Add & Norm
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class Encoder(nn.Module):
    def __init__(self, 
                 num_hidden_layers,            # block层数
                 hidden_size,                  # 隐藏层大小
                 hidden_act,                   # 隐藏层激活函数
                 intermediate_size,
                 num_attention_heads,          # 注意力头数量
                 attention_probs_dropout_prob, # attention prob 的 dropout 比例
                 hidden_dropout_prob,          # dropout 比例
                 layer_norm_eps,  
                ):
        super(Encoder, self).__init__()
        # PyTorch 中的 ModuleList https://zhuanlan.zhihu.com/p/64990232
        self.layer = nn.ModuleList([EncoderLayer(hidden_size, hidden_act, intermediate_size, 
                                                 num_attention_heads, attention_probs_dropout_prob, 
                                                 hidden_dropout_prob, layer_norm_eps,) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            # encode 部分是 self-attention ，qkv 来源一致
            hidden_states = layer_module(hidden_states, hidden_states, hidden_states, attention_mask)

        return hidden_states
