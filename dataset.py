import torch
from collections import Counter
from torchtext import data
import args as arg
from copy import copy

import ipdb


class PaperDataset:
    def __init__(self, args):
        args.path = args.datadir + args.data
        print("Loading Data from ", args.path)
        self.args = args
        self.make_vocab(args)

        print("\nVocab Sizes")
        for x in self.fields:
            try:
                print(x[0], len(x[1].vocab))
            except:
                try:
                    print(x[0], len(x[1].itos))
                except:
                    pass

    def build_ent_vocab(self, path):
        ents = ""

        with open(path, encoding='utf-8') as f:
            for l in f:
                ents += " " + l.split("\t")[1]

        itos = sorted(list(set(ents.split(" "))))
        itos[0] = "<unk>"
        itos[1] = "<pad>"

        stoi = {x: i for i, x in enumerate(itos)}
        return itos, stoi

    def make_graph(self, r, num_ent):
        """
        :param r: relation string of a dataset row
        :param num_ent: number of entities in a dataset row
        :return: adj, rel matrices
        """

        # convert triples to ent list with adj and rel matrices
        triples = r.strip().split(';')
        triple_list = [[int(y) for y in triple.strip().split()] for triple in triples]
        rel_list = [2]  # global root node (i.e. 'ROOT')
        adj_size = num_ent + 1 + (2 * len(triple_list))
        adj = torch.zeros(adj_size, adj_size)  # adjacency matrix (# Entity + Global root + # relations) ^2

        # connect all entities with global root node
        for i in range(num_ent):
            adj[i, num_ent] = 1
            adj[num_ent, i] = 1

        # each entities have self loop
        for i in range(adj_size):
            adj[i, i] = 1

        # parse all triples and add in adjacency matrix
        for triple in triple_list:
            rel_list.extend([triple[1] + 3, triple[1] + 3 + self.REL.size])  # extends with ['REL', 'REL_inv']
            head = triple[0]
            tail = triple[2]
            relation = num_ent + len(rel_list) - 2  # 'REL'
            relation_inverse = num_ent + len(rel_list) - 1  # 'REL_inv'
            adj[head, relation] = 1
            adj[relation, tail] = 1
            adj[tail, relation_inverse] = 1
            adj[relation_inverse, head] = 1

        rel_list = torch.LongTensor(rel_list)  # rel_list: ['ROOT', 'REL 1', 'REL 1_inv', 'REL 2', 'REL 2_inv', ...]

        return adj, rel_list

    def make_vocab(self, args):
        args.path = args.datadir + args.data
        self.INPUT = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>",
                                include_lengths=True)  # Title
        self.OUTPUT = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>",
                                 include_lengths=True)  # Gold Abstract, preprocessed
        self.TARGET = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>")
        self.ENT_TYPE = data.Field(sequential=True, batch_first=True, eos_token="<eos>")  # Entity Type
        self.ENT = data.RawField()  # Entity
        self.REL = data.RawField()  # Relation between entities
        self.SORDER = data.RawField()
        self.SORDER.is_target = False
        self.REL.is_target = False
        self.ENT.is_target = False
        self.fields = [("title", self.INPUT), ("ent", self.ENT), ("nerd", self.ENT_TYPE), ("rel", self.REL),
                       ("out", self.OUTPUT),
                       ("sorder", self.SORDER)]
        train = data.TabularDataset(path=args.path, format='tsv', fields=self.fields)

        print('Building Vocab... ', end='')

        # Output Vocab
        # mapping from generics to indices are at the last of the vocab
        # also includes indexed generics, (e.g. <method_0>) but in mixed order
        self.OUTPUT.build_vocab(train, min_freq=args.outunk)
        generics = ['<method>', '<material>', '<otherscientificterm>', '<metric>', '<task>']  # Entity Types
        self.OUTPUT.vocab.itos.extend(generics)
        for generic in generics:
            self.OUTPUT.vocab.stoi[generic] = self.OUTPUT.vocab.itos.index(generic)

        # Target Vocab
        # Same as Output Vocab, except for the indexed generics' indices
        # len(vocab) = 11738 / <method_0>, <material_0> ... : 11738, <method_1>, ... : 11739 and so on.
        self.TARGET.vocab = copy(self.OUTPUT.vocab)
        entity_types = ['method', 'material', 'otherscientificterm', 'metric', 'task']
        for entity_type in entity_types:
            for idx in range(40):
                s = "<" + entity_type + "_" + str(idx) + ">"
                self.TARGET.vocab.stoi[s] = len(self.TARGET.vocab.itos) + idx

        # Entity Type Vocab
        # Indices for not-indexed generics are same with those of output vocab
        self.ENT_TYPE.build_vocab(train, min_freq=0)
        for x in generics:
            self.ENT_TYPE.vocab.stoi[x] = self.OUTPUT.vocab.stoi[x]

        # Title Vocab
        self.INPUT.build_vocab(train, min_freq=args.entunk)

        # Relation Vocab
        # Adds relations.vocab + inverse of relations.vocab
        self.REL.special = ['<pad>', '<unk>', 'ROOT']
        with open(args.datadir + "/" + args.relvocab) as f:
            rel_vocab = [x.strip() for x in f.readlines()]
            self.REL.size = len(rel_vocab)
            rel_vocab += [x + "_inv" for x in rel_vocab]
            rel_vocab += self.REL.special
        self.REL.itos = rel_vocab

        self.ENT.itos, self.ENT.stoi = self.build_ent_vocab(args.path)

        print('Done')

        if not self.args.eval:
            self.make_iterator(train)

    def list_to(self, li):
        """Moves all parameters in li to device"""
        return [x.to(self.args.device) for x in li]

    def collate_fn(self, batch):

        ent, ent_len_list = zip(*batch.ent)  # batch.ent: (batch_size,) / list of x.ent
        # ent: (batch_size,) / tuple of tensors sized (num of entity, longest entity len) per each dataset row
        # ent_len_list: (batch_size,) / tuple of (num of entity) per each dataset row i.e. 각 row의 각 entity의 토큰 개수
        ent, ent_num_list = self.chunk_entity(ent)
        # ent: (sum of num of entities in each row, longest entity len)
        # ent_num_list: (batch_size,) / tensor of number of entities in each row
        ent = ent.to(self.args.device)
        ent_num_list = ent_num_list.to(self.args.device)

        ent_len_list = torch.cat(ent_len_list, 0).to(self.args.device)
        # ent_len_list: (sum of num of entities in each row, ) / tensor of 각 entity의 단어 개수

        batch.ent = (ent, ent_len_list, ent_num_list)

        adj, rel = zip(*batch.rel)  # batch.rel: (batch_size,) / list of x.rel
        # adj: (batch_size,) / tuple of adjacency matrices per each dataset row
        # rel: (batch_size,) / tuple of list of relations per each dataset row

        batch.rel = [self.list_to(adj), self.list_to(rel)]  # [[adj1, adj2, ...], [rel1, rel2, ...]]

        return batch

    def chunk_entity(self, ent):
        """
        Chunks entities in a batch into one tensor, and returns with 'each row's number of entity' tensor
        """
        lens = [x.size(0) for x in ent]  # lens: list of # of entities in each dataset row
        m = max([x.size(1) for x in ent])  # m: longest entity len in the batch
        data = [self.pad(x.transpose(0, 1), m).transpose(0, 1) for x in ent]
        data = torch.cat(data, 0)
        return data, torch.LongTensor(lens)  # data: pads each x in adj to (,m), and concat them

    def make_iterator(self, train):
        args = self.args
        t1, t2, t3 = [], [], []

        # Segregate training data by its output length
        for row in train:
            out_len = len(row.out)
            if out_len < 100:
                t1.append(row)
            elif 100 < out_len < 220:
                t2.append(row)
            else:
                t3.append(row)
        t1_dataset = data.Dataset(t1, self.fields)
        t2_dataset = data.Dataset(t2, self.fields)
        t3_dataset = data.Dataset(t3, self.fields)
        valid = data.TabularDataset(path=args.path.replace("train", "val"), format='tsv', fields=self.fields)

        print("Dataset Sizes (t1, t2, t3, valid):", end=' ')
        for dataset in [t1_dataset, t2_dataset, t3_dataset, valid]:
            print(len(dataset.examples), end=' ')

            for row in dataset:
                row.rawent = row.ent.split(" ; ")
                row.ent = self.vectorize_entity(row.ent, self.ENT)
                # row.ent: tuple of ((# of entities in x, max entity len), (# of entities))

                row.rel = self.make_graph(row.rel, len(row.ent[1]))

                row.tgt = row.out
                row.out = [y.split("_")[0] + ">" if "_" in y else y for y in row.out]
                # row.out: removes tag indices for out (e.g. <method_0> => <method>

            dataset.fields["tgt"] = self.TARGET
            dataset.fields["rawent"] = data.RawField()
            dataset.fields["rawent"].is_target = False

        self.t1_iter = data.Iterator(t1_dataset, args.t1size, device=args.device, sort_key=lambda x: len(x.out),
                                     repeat=False, train=True)
        self.t2_iter = data.Iterator(t2_dataset, args.t2size, device=args.device, sort_key=lambda x: len(x.out),
                                     repeat=False, train=True)
        self.t3_iter = data.Iterator(t3_dataset, args.t3size, device=args.device, sort_key=lambda x: len(x.out),
                                     repeat=False, train=True)
        self.val_iter = data.Iterator(valid, args.t3size, device=args.device, sort_key=lambda x: len(x.out),
                                      sort=False, repeat=False, train=False)

    def make_test(self, args):
        path = args.path.replace("train", 'test')
        fields = self.fields
        dataset = data.TabularDataset(path=path, format='tsv', fields=fields)

        for row in dataset:
            row.rawent = row.ent.split(" ; ")
            row.ent = self.vectorize_entity(row.ent, self.ENT)
            # x.ent: tuple of ((# of entities in x, max entity len), (# of entities))
            row.rel = self.make_graph(row.rel, len(row.ent[1]))
            # x.rel: tuple of (adj, rel)

            row.tgt = row.out
            row.out = [token.split("_")[0] + ">" if "_" in token else token for token in row.out]

        dataset.fields["tgt"] = self.TARGET
        dataset.fields["rawent"] = data.RawField()
        dataset.fields["rawent"].is_target = False

        test_iter = data.Iterator(dataset, 1, device=args.device, sort_key=lambda x: len(x.title), train=False,
                                  sort=False)
        return test_iter

    def vectorize_entity(self, ex, field):
        # returns tensor and lens
        ex = [[field.stoi[x] if x in field.stoi else 0 for x in y.strip().split(" ")] for y in ex.split(";")]
        return self.pad_list(ex, 1)

    def pad_list(self, l, ent=1):
        lens = [len(x) for x in l]
        m = max(lens)
        return torch.stack([self.pad(torch.tensor(x), m, ent) for x in l], 0), torch.LongTensor(lens)

    def pad(self, tensor, length, ent=1):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(ent)])

    def create_sentence(self, words, ents):
        ents = ents[0]
        vocab = self.TARGET.vocab
        string = ' '.join([vocab.itos[word] if word < len(vocab.itos)
                           else ents[word - len(vocab.itos)].upper()
                           for word in words])

        return string.split("<eos>")[0] if "<eos>" in string else string


if __name__ == "__main__":
    args = arg.parse_args()
    ds = PaperDataset(args)
    ipdb.set_trace()
