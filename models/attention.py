import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixAttention(nn.Module):

    def __init__(self, linin, linout):
        super().__init__()
        self.attnlin = nn.Linear(linin, linout)

    def forward(self, dec, emb):
        emb, elen = emb  # emb: (batch_size , max entity num, 500), elen: (batch_size,)
        emask = torch.arange(0, emb.size(1)).unsqueeze(0).repeat(emb.size(0), 1).long() #.cuda()
        emask = (emask >= elen.unsqueeze(1)).unsqueeze(1)  # (batch_size, max entity num)
        decsmall = self.attnlin(dec)  # (batch_size, max abstract len, 500)
        unnorm = torch.bmm(decsmall, emb.transpose(1, 2))  # (b_size, max abstract len, max entity num) / compute QK^T
        unnorm.masked_fill_(emask, -float('inf'))
        attn = F.softmax(unnorm, dim=2)
        out = torch.bmm(attn, emb)
        return out, attn  # attn: (b_size, max_abstract_len, max_entity_num) / entities attended on each abstract word


class MultiHeadAttention(nn.Module):
    def __init__(self, args, h=8, dropout_prob=0.5):
        super(MultiHeadAttention, self).__init__()

        self.query_dim = args.hsz
        self.val_dim = args.hsz
        self.key_dim = torch.tensor(args.hsz, requires_grad=False).float()
        self.h = h
        self.dropout_prob = dropout_prob

        self.query_layer = nn.Linear(self.query_dim, self.val_dim, bias=False)
        self.key_layer = nn.Linear(args.hsz, self.val_dim, bias=False)
        self.value_layer = nn.Linear(args.hsz, self.val_dim, bias=False)

    def forward(self, query, keys, mask=None):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        # e.g. Q: (27, 1, 500) => (108, 1, 125) / K: (27, 27, 500) => (108, 27, 125)
        size = int(self.val_dim / self.h)
        Q = torch.cat(Q.split(split_size=size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=size, dim=2), dim=0)

        # calculate QK^T
        # e.g. = (108, 1, 27)
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self.key_dim)

        if mask is not None:
            #  replace attention of entries without edge, with -inf
            mask = mask
            mask = mask.repeat(self.h, 1, 1)  # e.g. mask: (27, 1, 27) => (108, 1, 27)
            attention.masked_fill_(mask, -float('inf'))
        attention = F.softmax(attention, dim=-1)

        # apply dropout
        attention = F.dropout(attention, self.dropout_prob)

        # multiply it with V
        # e.g. = (108, 1, 125)
        attention = torch.matmul(attention, V)

        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self.h)
        attention = torch.cat(attention.split(split_size=restore_chunk_size, dim=0), dim=2)

        # residual connection
        attention += query  # (27, 1, 500)

        return attention  # V_caret
