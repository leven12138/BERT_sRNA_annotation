import torch
from torch import nn

from module.backbone_bert.package import BertConfig, BertOutput
from module.base_transformer.encoder import Encoder
from module.base_transformer.layers import InputEmbeddings


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 只取出第一个 token 也就是 cls 位置上的 embedding 进行 dense 变形
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, 
                 vocab_size,              # 字典长度
                 hidden_size,             # 隐藏层大小
                 hidden_act,
                 intermediate_size,
                 num_hidden_layers,
                 num_attention_heads,
                 max_position_embeddings, # 最长输入长度
                 attention_probs_dropout_prob,
                 hidden_dropout_prob,     # dropout 比例
                 layer_norm_eps,
                 initializer_range):
        super(BertModel, self).__init__()
        self.initializer_range = initializer_range
        self.embeddings = InputEmbeddings(vocab_size=vocab_size,
                                          hidden_size=hidden_size,
                                          max_position_embeddings=max_position_embeddings,
                                          hidden_dropout_prob=hidden_dropout_prob,
                                          layer_norm_eps=layer_norm_eps)
        self.encoder = Encoder(num_hidden_layers=num_hidden_layers,
                               hidden_size=hidden_size,
                               hidden_act=hidden_act,
                               intermediate_size=intermediate_size,
                               num_attention_heads=num_attention_heads,
                               attention_probs_dropout_prob=attention_probs_dropout_prob,
                               hidden_dropout_prob=hidden_dropout_prob,
                               layer_norm_eps=layer_norm_eps)
        self.pooler = BertPooler(hidden_size)

        self.init_weights()
        self.eval()

    @classmethod
    def from_config(cls, config_path):
        import json
        with open(config_path, "r", encoding='utf-8') as reader:
            config = json.loads(reader.read())
        return cls(**config)
        
    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 4D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]

        # 使用 dataloader 的时候纬度可能出问题 todo @mmmwhy
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        sequence_output = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(sequence_output)

        outputs = BertOutput(last_hidden_state=sequence_output, pooler_output=pooled_output)

        return outputs
