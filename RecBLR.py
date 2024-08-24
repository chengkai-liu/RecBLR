import torch
from torch import nn
from parallel_scan import parallel_scan
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch.nn.functional as F
import math
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None
from einops import rearrange

def softplus_inverse(x):
    return torch.log(torch.exp(x) - 1)


class RecBLR(SequentialRecommender):
    def __init__(self, config, dataset):
        super(RecBLR, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.expand = config["expand"]
        self.d_conv = config["d_conv"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
            
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.recurrent_layers = nn.ModuleList([
            RecurrentLayer(
                d_model=self.hidden_size,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.layer_norm(item_emb)
        
        for i in range(self.num_layers):
            item_emb = self.recurrent_layers[i](item_emb)
        
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
    
class RecurrentLayer(nn.Module):
    def __init__(self, d_model, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.behavior_modeling = GatedRecurrentLayer(
                d_model=d_model, 
                expansion_factor=expand, 
                kernel_size=d_conv
            )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
    
    def forward(self, input_tensor):
        hidden_states = self.behavior_modeling(input_tensor)
        hidden_states = self.layer_norm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states

    
class GatedRecurrentLayer(nn.Module):
    def __init__(self, d_model=64, expansion_factor=2, kernel_size=4):
        super().__init__()
        log_1 = math.log(1)
        r_min, r_max = 0.9, 0.999
        l_value = -math.log(r_min) / math.exp(log_1)
        r_value = -math.log(r_max) / math.exp(log_1)
        l = softplus_inverse(torch.tensor(l_value)).item()
        r = softplus_inverse(torch.tensor(r_value)).item()


        hidden = int(d_model * expansion_factor)
        self.input = nn.Linear(d_model, 2*hidden, bias=False)
        self.conv1d = nn.Conv1d(in_channels=hidden, out_channels=hidden, bias=True,
                              kernel_size=kernel_size, groups=hidden, padding=kernel_size-1)
        self.gates = nn.Linear(hidden, 2*hidden, bias=True)
        self.Lambda = nn.Parameter(torch.linspace(l, r, hidden))
        self.output = nn.Linear(hidden, d_model, bias=False)


    def forward(self, x):
        _, seq_len, _ = x.shape
        
        xz = self.input(x)
        x, z = xz.chunk(2, dim=-1)
        
        # pad to power of two (left padding)
        pad_len = 2 ** ((seq_len - 1).bit_length()) - seq_len
        if pad_len:
            x = F.pad(x, (0, 0, pad_len, 0), mode='constant', value=0)
        
        # temporal conv1d with causal padding
        if causal_conv1d_fn is None:
            x = F.silu(self.conv1d(x.mT)[..., :seq_len+pad_len].mT)
        else:
        # temporal conv1d with CUDA optimization
            x = causal_conv1d_fn(
                    x=x.mT,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation="silu",
                ).mT

        # behavior-dependent linear recurrent unit (BD-LRU)
        recurrence, input = self.gates(x).chunk(2, dim=-1)
        alpha = torch.exp(-F.softplus(self.Lambda) * torch.sigmoid(recurrence))
        beta = torch.sqrt(1 - alpha.pow(2) + 1e-8) * torch.sigmoid(input)
        beta_prime = beta  * x
        h = parallel_scan(alpha.mT.contiguous(), beta_prime.mT.contiguous()).mT

        # truncate to original sequence length
        if pad_len:
            h = h[:, pad_len:]
            
        x = self.output(F.silu(z) * h)
        return x
    
    
class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states
    
    
