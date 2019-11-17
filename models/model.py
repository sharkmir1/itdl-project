import torch
from torch import nn
from models.attention import MultiHeadAttention, SingleHeadAttention
from models.sequence_encoder import EntityEncoder, LSTMEncoder
from models.graph_encoder import GraphEncoder
from models.beam import Beam

import ipdb


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cat_times = 3 if args.title else 2
        self.embed = nn.Embedding(args.output_vocab_size, args.hsz)
        self.entity_encoder = EntityEncoder(args)
        self.graph_encoder = GraphEncoder(args)

        if args.title:
            self.title_encoder = LSTMEncoder(args, toks=args.input_vocab_size)
            self.attention_title = MultiHeadAttention(args, h=4, dropout_prob=args.drop)
            # attention_title: computes c_s (decoding-phase context vector for title)

        self.decoder = nn.LSTMCell(args.hsz * cat_times, args.hsz)
        self.attention_graph = MultiHeadAttention(args, h=4, dropout_prob=args.drop)
        # attention_graph: compute c_g (decoding-phase context vector for graph)

        self.mat_attention = SingleHeadAttention(args.hsz * cat_times, args.hsz, args.device)
        self.switch = nn.Linear(args.hsz * cat_times, 1)
        self.out = nn.Linear(args.hsz * cat_times, args.target_vocab_size)

    def forward(self, batch):
        if self.args.title:
            title_encoding, _ = self.title_encoder(batch.title)  # (batch_size, title_len, 500)
            title_mask = self.create_mask(title_encoding.size(), batch.title[1]).unsqueeze(1)  # (batch_size, 1, title_len)
        output, _ = batch.out  # (batch_size, max abstract len) / all tag indices removed
        ents = batch.ent  # tuple of (ent, ent_len_list, ent_num_list) / refer to collate_fn in dataset.py
        ent_num_list = ents[2]
        ents = self.entity_encoder(ents)
        # ents: (batch_size (num of rows), max entity num, 500) / encoded hidden states of each entity in batch

        glob, node_embeddings, node_mask = self.graph_encoder(batch.rel[0], batch.rel[1], (ents, ent_num_list))
        hx = glob  # this is the initial hidden state for the decoding LSTM
        node_mask = node_mask == 0  # (batch_size, max adj_matrix size) / 1 on relation vertices, else 0
        node_mask = node_mask.unsqueeze(1)

        cx = torch.tensor(hx)  # (batch_size, 500)
        context = torch.zeros_like(hx)
        if self.args.title:
            title_context = self.attention_title(hx.unsqueeze(1), title_encoding, mask=title_mask).squeeze(1)
            # (batch_size, 500) / c_s, context vector for each title
            context = torch.cat((context, title_context), 1)
            # (batch_size, 1000) / cat of c_g, c_s (c_g computed below)

        e = self.embed(output).transpose(0, 1)  # (max abstract len, batch_size, 500)
        outputs = []
        for i, k in enumerate(e):
            # k: (batch_Size, 500) / i_th keyword's embedding in the target abstract

            # for teacher-forcing, always include gold abstract word as prev in training
            prev = torch.cat((context, k), 1)  # (batch_size, 1500)
            hx, cx = self.decoder(prev, (hx, cx))  # compute new hx, cx with current (hx, cx, input abstract word)
            context = self.attention_graph(hx.unsqueeze(1), node_embeddings, mask=node_mask).squeeze(1)
            # graph_context: (batch_size, 500) / c_g, context vector for each graph (E6, E7)
            if self.args.title:
                title_context = self.attention_title(hx.unsqueeze(1), title_encoding, mask=title_mask).squeeze(1)
                # title_context: (batch_size, 500) / c_s, context vector for each title
                context = torch.cat((context, title_context), 1)
                # whole context: (batch_size, 1000) / cat of c_g, c_s

            out = torch.cat((hx, context), 1)  # (batch_size, 1500) / [ h_t || c_t ]
            outputs.append(out)

        outputs = torch.stack(outputs, 1)  # (batch_size, max abstract len, 1500)
        sampling_prob = torch.sigmoid(self.switch(outputs))
        # (batch_size, max abstract len, 1) / computes 1-p (E8)

        out = self.out(outputs)  # (batch_size, max abstract len, target vocab size)
        out = torch.softmax(out, 2)
        out = sampling_prob * out

        # compute copy attention
        z = self.mat_attention(outputs, (ents, ent_num_list))
        # z: (batch_size, max_abstract_len, max_entity_num) / entities attended on each abstract word, then softmaxed
        z = (1 - sampling_prob) * z
        out = torch.cat((out, z), 2)
        out = out + (1e-6 * torch.ones_like(out))  # add epsilon to avoid underflow
        return out.log(), z

    def create_mask(self, size, l):
        """
        :param size: (batch_size, title_len, 500)
        :param l: tensor of (batch_size,) / lengths of each example
        :return: (batch_size, 1, title_len) / 각 row의 길이 + 1보다 더 큰 부분만 1, else 0
        """
        mask = torch.arange(0, size[1]).unsqueeze(0).repeat(size[0], 1)
        mask = mask.long().to(self.args.device)
        mask = (mask <= l.unsqueeze(1))
        mask = mask == 0
        return mask

    def trim_entity_index(self, output, vertex):
        """
        e.g. assume output: [[3, 100, 11744, 3]] / output_vocab_size: 11738 / vertex: [~, ~, ~, ~, ~, ~, 11733, ~]
        11744 accords to word among [<method_6>, ... <task_6>] (i.e. it is 6th entity in this row),
        and in vertex, the 6th entity in this row accords to 11733. (which is <method>)

        then function returns [[3, 100, 11733, 3]] as output.

        :param output: (beam_size,) / selected word for each beam (including indexed entities e.g. <method_0>)
        :param vertex: (1, num of entity in this row) /
                       indices of not-indexed entities (same as that in output vocab) + <eos>
        :return: output with all indexed tags changed to not-indexed tags (only types)
        """
        mask = output >= self.args.output_vocab_size  # 1 if words are indexed tags (e.g. <method_0>)
        if mask.sum() > 0:
            idxs = (output - self.args.output_vocab_size)
            idxs = idxs[mask]
            verts = vertex.index_select(1, idxs)
            output.masked_scatter_(mask, verts)

        # if words all in output vocab, return same outp
        return output

    def beam_generate(self, batch, beam_size, k):
        if self.args.title:
            title_encoding, _ = self.title_encoder(batch.title)  # (1, title_len, 500)
            title_mask = self.create_mask(title_encoding.size(), batch.title[1]).unsqueeze(1)  # (1, 1, title_len)

        ents = batch.ent
        ent_num_list = ents[2]
        ents = self.entity_encoder(ents)
        # ents: (1, entity num, 500) / encoded hidden state of entities in b

        glob, node_embeddings, node_mask = self.graph_encoder(batch.rel[0], batch.rel[1], (ents, ent_num_list))
        hx = glob
        node_mask = node_mask == 0
        node_mask = node_mask.unsqueeze(1)

        cx = torch.tensor(hx)  # (beam size, 500)
        context = self.attention_graph(hx.unsqueeze(1), node_embeddings, mask=node_mask).squeeze(1)
        # (1, 500) / c_g
        if self.args.title:
            title_context = self.attention_title(hx.unsqueeze(1), title_encoding, mask=title_mask).squeeze(1)
            # (1, 500) / c_s
            context = torch.cat((context, title_context), 1)  # (beam size, 1000) / cat of c_g, c_s

        recent_token = torch.LongTensor(ents.size(0), 1).fill_(self.starttok)  # initially, (1, 1) / start token
        beam = None
        for i in range(self.args.maxlen):
            op = self.trim_entity_index(recent_token.clone(), batch.nerd)
            # previous token을 만들기 위해 tag로부터 index를 없애는 작업
            # 없애는 이유: train할 때도 output vocab (input) => target vocab (output) 으로 train 되었기 때문.
            op = self.embed(op).squeeze(1)  # (beam size, 500)
            prev = torch.cat((context, op), 1)  # (beam size, 1000)
            hx, cx = self.decoder(prev, (hx, cx))
            context = self.attention_graph(hx.unsqueeze(1), node_embeddings, mask=node_mask).squeeze(1)
            if self.args.title:
                title_context = self.attention_title(hx.unsqueeze(1), title_encoding, mask=title_mask).squeeze(1)
                context = torch.cat((context, title_context), 1)  # (beam size, 1000)

            out = torch.cat((hx, context), 1).unsqueeze(1)  # (beam size, 1, 1500)
            sampling_prob = torch.sigmoid(self.switch(out))  # (beam size, 1, 1)

            out = self.out(out)  # (beam size, 1, target vocab size)
            out = torch.softmax(out, 2)
            out = sampling_prob * out

            # compute copy attention
            z = self.mat_attention(out, (ents, ent_num_list))
            # z: (1,  max_abstract_len, max_entity_num) / entities attended on each abstract word, then softmaxed

            z = (1 - sampling_prob) * z
            out = torch.cat((out, z), 2)  # (beam size, 1, target vocab size + entity num)
            out[:, :, 0].fill_(0)  # remove probability for special tokens <unk>, <init>
            out[:, :, 1].fill_(0)

            out = out + (1e-6 * torch.ones_like(out))
            decoded = out.log()
            scores, words = decoded.topk(dim=2, k=k)  # (beam size, 1, k), (beam size, 1, k)
            if not beam:
                beam = Beam(words.squeeze(), scores.squeeze(), [hx] * beam_size,
                            [cx] * beam_size, [context] * beam_size,
                            beam_size, k, self.args.output_vocab_size, self.args.device)
                beam.endtok = self.endtok
                beam.eostok = self.eostok
                node_embeddings = node_embeddings.repeat(len(beam.beam), 1, 1)  # (beam size, adjacency matrix len, 500)
                node_mask = node_mask.repeat(len(beam.beam), 1, 1)  # (beam size, 1, adjacency matrix len) => all 0?
                if self.args.title:
                    title_encoding = title_encoding.repeat(len(beam.beam), 1, 1)  # (beam size, title_len, 500)
                    title_mask = title_mask.repeat(len(beam.beam), 1, 1)  # (1, 1, title_len)

                ents = ents.repeat(len(beam.beam), 1, 1)  # (beam size, entity num, 500)
                ent_num_list = ent_num_list.repeat(len(beam.beam))  # (beam size,)
            else:
                # if all beam nodes have ended, stop generating
                if not beam.update(scores, words, hx, cx, context):
                    break
                # if beam size changes (i.e. any of beam ends), change size of weight matrices accordingly
                node_embeddings = node_embeddings[:len(beam.beam)]
                node_mask = node_mask[:len(beam.beam)]
                if self.args.title:
                    title_encoding = title_encoding[:len(beam.beam)]
                    title_mask = title_mask[:len(beam.beam)]
                ents = ents[:len(beam.beam)]
                ent_num_list = ent_num_list[:len(beam.beam)]
            recent_token = beam.getwords()  # (beam size,) / next word for each beam
            hx = beam.get_h()  # (beam size, 500)
            cx = beam.get_c()  # (beam size, 500)
            context = beam.get_context()  # (beam size, 1000)

        return beam
