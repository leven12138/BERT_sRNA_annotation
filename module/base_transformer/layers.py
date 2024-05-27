"""
与 transformer 相关各层的实现，包括:
1、InputEmbeddings, 用于实现 Positional Encoding 和 Input Embedding 部分，Output Embedding 与这个应该是完全一致的。
2、MultiHeadAttentionLayer, 用于实现 Mutli-Head Attention。
3、AddNorm, 用于实现 Add & Norm 部分。
4、HalfFeedForward, 实现了 FFN(x) = max(0, xW1 + b1)W2 + b2 中的 max(0, xW1 + b1)W2 部分。需要与 AddNorm 配合使用。
5、AttentionAddNorm, 将 Multi-Head Attention 和 Add & Norm 合并在一起，原始 bert 是这么写的，可能是因为这两东西经常一起出现。
"""

import math
import torch

from torch import nn
from module.common.activate import activations


class InputEmbeddings(nn.Module):
    def __init__(self, 
                 vocab_size,              # 字典长度
                 hidden_size,             # 隐藏层大小
                 max_position_embeddings, # 最长输入长度
                 hidden_dropout_prob,     # dropout 比例
                 layer_norm_eps,          
                 type_vocab_size=None,    # 输入中最多包含句子数量
                 ext_vocab_size=None):
        super(InputEmbeddings, self).__init__()

        vocab_size = ext_vocab_size if ext_vocab_size else vocab_size  # 给定 ext_vocab_size 的时候走 ext
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        if type_vocab_size:
            self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings  # 注意按位相加

        # 有些任务不需要区别 token_type
        if hasattr(self, "token_type_embeddings"):
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, 
                 hidden_size,                 # 隐藏层大小
                 num_attention_heads,         # 注意力头数量
                 attention_probs_dropout_prob # attention prob 的 dropout 比例
                ): 
        super(MultiHeadAttentionLayer, self).__init__()

        assert hidden_size % num_attention_heads == 0, "隐藏层纬度 需为 注意力头的数量 整数倍，否则注意力 embedding 无法计算"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask=None):
        """
        query shape: [batch_size, query_len, hidden_size]
        key shape: [batch_size, key_len, hidden_size]
        value shape: [batch_size, value_len, hidden_size]
        在 bert 中，query_len、key_len、value_len 三者相等
        """

        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)
        """
        mixed_query_layer shape: [batch_size, query_len, hidden_size]
        mixed_query_layer shape: [batch_size, key_len, hidden_size]
        mixed_query_layer shape: [batch_size, value_len, hidden_size]
        """

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        """
        query_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        key_layer shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        value_layer shape: [batch_size, num_attention_heads, value_len, attention_head_size]
        """

        # 交换 k 的最后两个维度，然后 q 和 k 执行点积, 获得 attention score
        # attention_scores shape: [batch_size, num_attention_heads, query_len, key_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # attention_mask 的值是 -inf, softmax 后的权重就是 0 了
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 对注意力结果进行 softmax， 得到 query 对于每个 value 的 score
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 注意这里的实现是比较特别的，他是把某个 value 的 score 整个 mask 掉，但原始论文的确是这个意思
        # 这里引出一个很有趣的预训练方式，我们使用两个权重完全相同的 bert 进行对比学习 (比如搞 moco )，而可行的原因就是 drop 不一致
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]

        # transpose、permute 等维度变换操作后，tensor 在内存中不再是连续存储的，而 view 操作要求 tensor 的内存连续存储，
        # 所以在调用 view 之前，需要 contiguous 来返回一个 contiguous copy；
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer shape: [batch_size, query_len, num_attention_heads, attention_head_size]

        # 注意这里又把最后两个纬度合回去了，做的是 view 操作
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        outputs = context_layer.view(*new_context_layer_shape)

        return outputs


class AddNorm(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob, layer_norm_eps):
        super(AddNorm, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 残差，非常重要
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class HalfFeedForward(nn.Module):
    def __init__(self, 
                 hidden_size,
                 intermediate_size,
                 hidden_act):
        super(HalfFeedForward, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = activations[hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class AttentionAddNorm(nn.Module):
    def __init__(self, 
                 hidden_size,                  # 隐藏层大小
                 num_attention_heads,          # 注意力头数量
                 attention_probs_dropout_prob, # attention prob 的 dropout 比例
                 hidden_dropout_prob,          # dropout 比例
                 layer_norm_eps,          
                ): 
        super(AttentionAddNorm, self).__init__()
        self.self = MultiHeadAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = AddNorm(hidden_size, hidden_size, hidden_dropout_prob, layer_norm_eps)

    def forward(self, query_tensor, key_tensor, value_tensor, attention_mask=None):
        # self attention 中 query_tensor, key_tensor, value_tensor 是一致的
        self_outputs = self.self(query_tensor, key_tensor, value_tensor, attention_mask)
        attention_output = self.output(self_outputs, query_tensor)
        return attention_output
