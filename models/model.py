import torch
from torch import nn
from models.attention import MultiHeadAttention, MatrixAttention
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

        self.lstm = nn.LSTMCell(args.hsz * cat_times, args.hsz)
        self.entity_encoder = EntityEncoder(args)
        self.switch = nn.Linear(args.hsz * cat_times, 1)
        self.attention = MultiHeadAttention(args.hsz, args.hsz, args.hsz, h=4, dropout_p=args.drop)
        self.mat_attention = MatrixAttention(args.hsz * cat_times, args.hsz)
        self.graph_encoder = GraphEncoder(args)

        if args.title:
            self.title_encoder = LSTMEncoder(args, toks=args.input_vocab_size)
            self.attention_title = MultiHeadAttention(args.hsz, args.hsz, args.hsz, h=4, dropout_p=args.drop)
            # attn2: computes c_s (decoding-phase context vector for title)

        self.out = nn.Linear(args.hsz * cat_times, args.target_vocab_size)

    def forward(self, batch):
        if self.args.title:
            title_embedding, _ = self.title_encoder(batch.title)  # (batch_size, title_len, 500)
            title_mask = self.create_mask(title_embedding.size(), batch.title[1]).unsqueeze(1)  # (batch_size, 1, title_len)
        outp, _ = batch.out  # (batch_size, max abstract len) / all tag indices removed
        ents = batch.ent  # tuple of (ent, ent_len_list, ent_num_list) / refer to collate_fn in dataset.py
        ent_num_list = ents[2]
        ents = self.entity_encoder(ents)
        # ents: (batch_size (num of rows), max entity num, 500) / encoded hidden states of each entity in batch

        if self.graph:
            gents, glob, grels = self.graph_encoder(batch.rel[0], batch.rel[1], (ents, ent_num_list))
            hx = glob  # this is the initial hidden state for decoding lstm
            keys, mask = grels
            mask = mask == 0  # (batch_size, max adj_matrix size) / 1 on relation vertices, else 0
        else:
            mask = self.create_mask(ents.size(), ent_num_list)
            hx = ents.mean(dim=1)
            keys = ents
        mask = mask.unsqueeze(1)
        planlogits = None

        cx = torch.tensor(hx)  # (batch_size, 500)
        a = torch.zeros_like(hx)
        if self.args.title:
            a2 = self.attention_title(hx.unsqueeze(1), title_embedding, mask=title_mask).squeeze(1)
            # a2: (batch_size, 500) / c_s, context vector for each title
            a = torch.cat((a, a2), 1)
            # a: (batch_size, 1000) / cat of c_g, c_s (c_g computed below)

        e = self.embed(outp).transpose(0, 1)  # (max abstract len, batch_size, 500)
        outputs = []
        for i, k in enumerate(e):
            # k: (batch_Size, 500) / i_th keyword's embedding in the target abstract

            prev = torch.cat((a, k), 1)  # (batch_size, 1500)
            hx, cx = self.lstm(prev, (hx, cx))  # compute new hx, cx with current (hx, cx, input abstract word)
            a = self.attention(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
            # a: (batch_size, 500) / c_g, context vector for each graph (equation 6,7 in paper)
            if self.args.title:
                a2 = self.attention_title(hx.unsqueeze(1), title_embedding, mask=title_mask).squeeze(1)
                # a2: (batch_size, 500) / c_s, context vector for each title
                a = torch.cat((a, a2), 1)
                # a: (batch_size, 1000) / cat of c_g, c_s (c_g computed below)

            out = torch.cat((hx, a), 1)  # (batch_size, 1500) / [ h_t || c_t ] in paper
            outputs.append(out)
        l = torch.stack(outputs, 1)  # (batch_size, max abstract len, 1500)
        s = torch.sigmoid(self.switch(l))  # (batch_size, max abstract len, 1) / 1-p (p in equation 8 in paper)
        o = self.out(l)  # (batch_size, max abstract len, target vocab size)
        o = torch.softmax(o, 2)
        o = s * o
        # compute copy attn
        _, z = self.mat_attention(l, (ents, ent_num_list))
        # z: (batch_size, max_abstract_len, max_entity_num) / entities attended on each abstract word, then softmaxed
        z = (1 - s) * z
        o = torch.cat((o, z), 2)
        o = o + (1e-6 * torch.ones_like(o))  # add epsilon to avoid underflow
        return o.log(), z, planlogits

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

    def emb_w_vertex(self, outp, vertex):
        """
        e.g. assume outp: [[3, 100, 11744, 3]] / output_vocab_size: 11738 / vertex: [~, ~, ~, ~, ~, ~, 11733, ~]
        11744 accords to word among [<method_6>, ... <task_6>] (i.e. it is 6th entity in this row),
        and in vertex, the 6th entity in this row accords to 11733. (which is <method>)

        then function returns [[3, 100, 11733, 3]] as output.

        :param outp: (beam_size,) / selected word for each beam (including indexed entities e.g. <method_0>)
        :param vertex: (1, num of entity in this row) /
                       indices of not-indexed entities (same as that in output vocab) + <eos>
        :return: output with all indexed tags changed to not-indexed tags (only types)
        """
        mask = outp >= self.args.output_vocab_size  # 1 if words are indexed tags (e.g. <method_0>)
        if mask.sum() > 0:
            idxs = (outp - self.args.ntoks)
            idxs = idxs[mask]
            verts = vertex.index_select(1, idxs)
            outp.masked_scatter_(mask, verts)

        # if words all in output vocab, return same outp
        return outp

    def beam_generate(self, b, beamsz, k):
        if self.args.title:
            tencs, _ = self.title_encoder(b.title)  # (1, title_len, 500)
            tmask = self.create_mask(tencs.size(), b.title[1]).unsqueeze(1)  # (1, 1, title_len)
        ents = b.ent  # tuple of (ent, phlens, elens)
        entlens = ents[2]
        ents = self.entity_encoder(ents)
        # ents: (1, entity num, 500) / encoded hidden state of entities in b

        if self.graph:
            gents, glob, grels = self.graph_encoder(b.rel[0], b.rel[1], (ents, entlens))
            hx = glob
            # hx = ents.max(dim=1)[0]
            keys, mask = grels
            mask = mask == 0
        else:
            mask = self.create_mask(ents.size(), entlens)
            hx = ents.max(dim=1)[0]
            keys = ents
        mask = mask.unsqueeze(1)
        planlogits = None

        cx = torch.tensor(hx)  # (beam size, 500)
        a = self.attention(hx.unsqueeze(1), keys, mask=mask).squeeze(1)  # (1, 500) / c_g
        if self.args.title:
            a2 = self.attention_title(hx.unsqueeze(1), tencs, mask=tmask).squeeze(1)  # (1, 500) / c_s
            a = torch.cat((a, a2), 1)  # (beam size, 1000) / c_t

        outp = torch.LongTensor(ents.size(0), 1).fill_(self.starttok).cuda()  # initially, (1, 1) / start token
        beam = None
        for i in range(self.maxlen):
            op = self.emb_w_vertex(outp.clone(), b.nerd)
            # tag로부터 index를 없애는 작업 => train할 때도 not indexed(input) => indexed(output)으로 train 되었기 때문.
            if self.args.plan:
                schange = op == self.args.dottok
                if schange.nonzero().size(0) > 0:
                    print(schange, planplace, sorder)
                    planplace[schange.nonzero().squeeze()] += 1
                    for j in schange.nonzero().squeeze(1):
                        if planplace[j] < len(sorder[j]):
                            mask[j] = 0
                            m = sorder[j][planplace[j]]
                            mask[j][0][sorder[j][planplace[j]]] = 1
            op = self.embed(op).squeeze(1)  # (beam size, 500)
            prev = torch.cat((a, op), 1)  # (beam size, 1000)
            hx, cx = self.lstm(prev, (hx, cx))
            a = self.attention(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
            if self.args.title:
                a2 = self.attention_title(hx.unsqueeze(1), tencs, mask=tmask).squeeze(1)
                a = torch.cat((a, a2), 1)  # (beam size, 1000)
            l = torch.cat((hx, a), 1).unsqueeze(1)  # (beam size, 1, 1500)
            s = torch.sigmoid(self.switch(l))  # (beam size, 1, 1)
            o = self.out(l)  # (beam size, 1, target vocab size)
            o = torch.softmax(o, 2)
            o = s * o
            # compute copy attn
            _, z = self.mat_attention(l, (ents, entlens))
            # z = torch.softmax(z,2)
            z = (1 - s) * z
            o = torch.cat((o, z), 2)  # (beam size, 1, target vocab size + entity num)
            o[:, :, 0].fill_(0)  # remove probability for special tokens <unk>, <init>
            o[:, :, 1].fill_(0)

            o = o + (1e-6 * torch.ones_like(o))
            decoded = o.log()
            scores, words = decoded.topk(dim=2, k=k)  # (beam size, 1, k), (beam size, 1, k)
            if not beam:
                beam = Beam(words.squeeze(), scores.squeeze(), [hx for i in range(beamsz)],
                            [cx for i in range(beamsz)], [a for i in range(beamsz)],
                            beamsz, k, self.args.output_vocab_size)
                beam.endtok = self.endtok
                beam.eostok = self.eostok
                keys = keys.repeat(len(beam.beam), 1, 1)  # (beam size, adjacency matrix len, 500)
                mask = mask.repeat(len(beam.beam), 1, 1)  # (beam size, 1, adjacency matrix len) => all 0?
                if self.args.title:
                    tencs = tencs.repeat(len(beam.beam), 1, 1)  # (beam size, title_len, 500)
                    tmask = tmask.repeat(len(beam.beam), 1, 1)  # (1, 1, title_len)
                if self.args.plan:
                    planplace = planplace.unsqueeze(0).repeat(len(beam.beam), 1)
                    sorder = sorder * len(beam.beam)

                ents = ents.repeat(len(beam.beam), 1, 1)  # (beam size, entity num, 500)
                entlens = entlens.repeat(len(beam.beam))  # (beam size,)
            else:
                if not beam.update(scores, words, hx, cx, a):
                    break
                # if beam size changes (i.e. any of beam ends), change size of weight matrices accordingly
                keys = keys[:len(beam.beam)]
                mask = mask[:len(beam.beam)]
                if self.args.title:
                    tencs = tencs[:len(beam.beam)]
                    tmask = tmask[:len(beam.beam)]
                if self.args.plan:
                    planplace = planplace[:len(beam.beam)]
                    sorder = sorder[0] * len(beam.beam)
                ents = ents[:len(beam.beam)]
                entlens = entlens[:len(beam.beam)]
            outp = beam.getwords()  # (beam size,) / next word for each beam
            hx = beam.geth()  # (beam size, 500)
            cx = beam.getc()  # (beam size, 500)
            a = beam.getlast()  # (beam size, 1000)

        return beam
