import torch

from torch import nn
from module.common.activate import activations
from module.backbone_bert.package import BertConfig
from module.backbone_bert.bert_model import BertModel

CLS_IDX, SEP_IDX, MASK_IDX = 0, 0, 0

class BertForLMTransformHead(nn.Module):
    def __init__(self, hidden_size, hidden_act, vocab_size, layer_norm_eps):
        super(BertForLMTransformHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = activations[hidden_act]
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.decoder.bias = nn.Parameter(torch.zeros(vocab_size))
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.transform_act_fn(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.LayerNorm(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.decoder(hidden_states)
        return hidden_states # [src_len, batch_size, vocab_size]

class BertForMaskedLM(nn.Module):
    def __init__(self, config_path, num_hidden_layers=1, num_attention_heads=4):
        super(BertForMaskedLM, self).__init__()
        self.config = BertConfig(config_path)

        hidden_size = int(self.config.hidden_size / self.config.num_attention_heads * num_attention_heads)
        intermediate_size = int(self.config.intermediate_size / self.config.num_attention_heads * num_attention_heads)

        self.bert = BertModel(vocab_size=self.config.vocab_size,
                              hidden_size=hidden_size,
                              hidden_act=self.config.hidden_act,
                              intermediate_size=intermediate_size,
                              num_hidden_layers=num_hidden_layers,
                              num_attention_heads=num_attention_heads,
                              max_position_embeddings=self.config.max_position_embeddings,
                              attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
                              hidden_dropout_prob=self.config.hidden_dropout_prob,
                              layer_norm_eps=self.config.layer_norm_eps,
                              initializer_range=self.config.initializer_range)
        self.vocab_size = self.config.vocab_size
        self.classifier = BertForLMTransformHead(hidden_size, self.config.hidden_act, self.config.vocab_size, self.config.layer_norm_eps)
        
    def forward(self,
                input_ids,            # [batch_size, src_len]
                attention_mask=None,  # [batch_size, src_len]
                token_type_ids=None,  # [batch_size, src_len]
                position_ids=None,
                masked_lm_labels=None # [batch_size, src_len]
                ):  
        all_encoder_outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids)
        
        sequence_output = all_encoder_outputs.last_hidden_state # [batch_size, src_len, hidden_size]
        prediction_scores = self.classifier(sequence_output) # [batch_size, src_len, vocab_size]
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.vocab_size),
                                      masked_lm_labels.reshape(-1))
            return masked_lm_loss, prediction_scores
        else:
            return prediction_scores