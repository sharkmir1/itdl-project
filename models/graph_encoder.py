import torch
from torch import nn
from models.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = MultiHeadAttention(args, h=4, dropout_prob=args.transformer_drop)
        self.linear1 = nn.Linear(args.hsz, args.hsz * 4)
        self.linear2 = nn.Linear(args.hsz * 4, args.hsz)
        self.layer_norm1 = nn.LayerNorm(args.hsz)
        self.layer_norm2 = nn.LayerNorm(args.hsz)
        self.dropout = nn.Dropout(args.drop)
        self.prelu = nn.PReLU(args.hsz * 4)

    def forward(self, q, k, m):
        q = self.attn(q, k, mask=m).squeeze(1)  # (adj_len(= # of entities + # of relations), 500)
        t = self.layer_norm1(q)
        q = self.linear2(self.prelu(self.linear1(t)))
        q = self.dropout(q)
        q = self.layer_norm2(q + t)
        return q  # V_tilde / (adj_len ( = # of entities + # of relations), 500)


class GraphEncoder(nn.Module):
    """
    Graph Transformer
    Partially referred to transformer implementation by HuggingFace (https://github.com/huggingface/transformers)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.relation_embed = nn.Embedding(args.relation_vocab_size, args.hsz)
        nn.init.xavier_normal_(self.relation_embed.weight)

        self.blocks = nn.ModuleList([TransformerBlock(args) for _ in range(args.transformer_num)])

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

    def forward(self, adjs, rels, ent_vec):
        """
        :param adjs: list of adjacency matrices in batch
        :param rels: list of tensors of relations per row in batch
        :param ent_vec: tuple (tensor(batch_size, max entity num, 500 ): encoded entities per row,
                            tensor(batch_size): # of entities in rows)
        :return: None, glob, (gents, emask)
                 global_node_embed: (batch_size, 500) / global node embedding
                 node_embeddings: (batch_size, max adj_matrix size, 500) / graph contextualized vertex encodings
                 mask: (batch_size, max adj_matrix size) / 각 adj_matrix의 size + 1보다 작은 부분만 1
        """
        ent_vec, entity_num = ent_vec

        rel_vec = [self.relation_embed(x) for x in rels]  # list of (num of relations, 500) per row in batch
        global_node_vec = []
        graphs = []

        for i, adj in enumerate(adjs):
            query_graph = torch.cat((ent_vec[i][:entity_num[i]], rel_vec[i]), 0)
            # query_graph: (# of entities + # of relations, 500) / cat(embedding(entities), embedding(relations))
            size = query_graph.size(0)  # adj_matrix size
            mask = (adj == 0).unsqueeze(1)

            for j in range(self.args.transformer_num):
                key_graph = torch.tensor(query_graph.repeat(size, 1).view(size, size, -1), requires_grad=False)
                # key_graph: (N, N, 500) / cat of query_graph N times
                query_graph = self.blocks[j](query_graph.unsqueeze(1), key_graph, mask)  # (N, 500)

            graphs.append(query_graph)
            global_node_vec.append(query_graph[entity_num[i]])  # append each global node's embedding
        global_node_vec = torch.stack(global_node_vec, 0)  # (batch_size, 500)

        graph_size_vec = [x.size(0) for x in graphs]
        node_embeddings = torch.stack([self.pad(x, max(graph_size_vec)) for x in graphs], 0)  # (batch_size, max N, 500)
        graph_size_vec = torch.LongTensor(graph_size_vec).to(self.args.device)

        mask = torch.arange(0, node_embeddings.size(1)).unsqueeze(0).repeat(node_embeddings.size(0), 1)
        mask = mask.long().to(self.args.device)
        mask = (mask <= graph_size_vec.unsqueeze(1))  # (batch_size, max N) / 각 adj_matrix의 size + 1보다 작은 부분만 1

        return global_node_vec, node_embeddings, mask
