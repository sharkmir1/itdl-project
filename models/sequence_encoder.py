import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EntityEncoder(nn.Module):
    """
    Encodes list of each row's `entity list` into sequence of 500-dim vector per entity
    """
    def __init__(self, args):
        super().__init__()
        self.lstm_encoder = LSTMEncoder(args, toks=args.entity_vocab_size)

    def forward(self, ents):
        ents, ent_len_list, ent_num_list = ents
        ent_num_list = ent_num_list.tolist()

        # feed in each entity (with multi tokens) into LSTM Encoder
        _, h = self.lstm_encoder((ents, ent_len_list))  # hidden states of LSTM: (num of entities, 4, 250)
        h = h[:, 2:]  # (batch_size, 2, 250)
        h = torch.cat([h[:, i] for i in range(h.size(1))], 1)  # enc: (num of entities, 500) / last layer hidden state

        # split chunked batch into tensors of each dataset row
        max_num = max(ent_num_list)  # max number of entities in a row
        encs = [self.pad(ent_embedding, max_num) for ent_embedding in h.split(tuple(ent_num_list))]
        out = torch.stack(encs, 0)

        return out  # (batch_size, max_num, 500)

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])


class LSTMEncoder(nn.Module):
    def __init__(self, args, toks=None):
        super().__init__()

        self.embed = nn.Embedding(toks, args.esz)
        self.dropout = nn.Dropout(args.embdrop)
        self.lstm = nn.LSTM(args.esz, args.hsz // 2, bidirectional=True, num_layers=args.layers, batch_first=True)

        nn.init.xavier_normal_(self.embed.weight)

    def forward(self, inp):
        sent, sent_len_list = inp  # l: (batch_size, title_len), ilens: (batch_size) / list of lengths of each title
        emb = self.embed(sent)
        emb = self.dropout(emb)

        sent_lens, indices = sent_len_list.sort(descending=True)
        e = emb.index_select(0, indices)
        e = pack_padded_sequence(e, sent_lens, batch_first=True)
        e, (h, c) = self.lstm(e)
        e, _ = pad_packed_sequence(e, batch_first=True)
        e = torch.zeros_like(e).scatter(0, indices.unsqueeze(1).unsqueeze(1).expand(-1, e.size(1), e.size(2)), e)
        h = h.transpose(0, 1)
        h = torch.zeros_like(h).scatter(0, indices.unsqueeze(1).unsqueeze(1).expand(-1, h.size(1), h.size(2)), h)

        # e: (batch_size, title_len, hsz) / h: (batch_size, layer_size * 2 (=4), hsz//2)
        return e, h
