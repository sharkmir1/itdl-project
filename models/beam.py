import torch


"""
Referred to https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py
for implementing Beam Search
"""


class BeamNode:
    def __init__(self, init_word, init_score, h, c, context):
        self.words = [init_word]  # each node contains one starting word (after init token)
        self.score = init_score
        self.h = h
        self.c = c
        self.context = context
        self.init_words = [init_word]
        self.prevent = 0
        self.just_started = False


class Beam:
    def __init__(self, words, scores, hs, cs, context, beam_size, k, vocab_size, device):
        """
        :param words: (k) / top k word indices
        :param scores: (k) / top k words scores
        :param hs: [hx (1, 500)] * beam_size
        :param cs: [cx (1, 500)] * beam_size
        :param context: [c_t (1,1000)] * beam size
        :param beam_size: default 4
        :param k: default 6
        :param vocab_size: output vocab size (without entities, but including tags-both indexed and not indexed)        
        """
        self.k = k
        self.beam = []  # list containing beam nodes
        for i in range(beam_size):
            self.beam.append(BeamNode(words[i].item(), scores[i].item(), hs[i], cs[i], context[i]))
        self.finished_nodes = []
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.device = device

    def sort(self, norm=True):
        if len(self.finished_nodes) < self.beam_size:
            self.finished_nodes.extend(self.beam)
            self.finished_nodes = self.finished_nodes[:self.beam_size]
        if norm:
            self.finished_nodes = sorted(self.finished_nodes, key=lambda x: x.score / len(x.words), reverse=True)
        else:
            self.finished_nodes = sorted(self.finished_nodes, key=lambda x: x.score, reverse=True)

    def copy_node(self, node):
        new_node = BeamNode(None, None, None, None, None)
        new_node.words = [word for word in node.words]
        new_node.score = node.score
        new_node.prevent = node.prevent
        new_node.init_words = [word for word in node.firstwords]
        new_node.just_started = node.isstart
        return new_node

    def getwords(self):
        """
        :return: last word of each beam
        """
        return torch.tensor([[x.words[-1]] for x in self.beam]).long().to(self.device)

    def get_h(self):
        return torch.cat([x.h for x in self.beam], dim=0)

    def get_c(self):
        return torch.cat([x.c for x in self.beam], dim=0)

    def get_context(self):
        return torch.cat([x.context for x in self.beam], dim=0)

    def get_scores(self):
        """
        :return: (beam size, k) / each row's entries are same (current scores)
        """
        return torch.tensor([[x.score] for x in self.beam]).float() \
            .repeat(1, self.k).to(self.device)

    def update(self, scores, words, hs, cs, context):
        """
        :param scores: (beam size, 1, k) / top k word scores for each beam
        :param words: (beam size, 1, k) / top k word indices for each beam
        :param hs: (beam size, 500)
        :param cs: (beam size, 500)
        :param context: (beam size, 1000)
        :return:
        """
        beam = self.beam
        scores = scores.squeeze()  # (beam size, k)
        words = words.squeeze()
        k = self.k
        scores = scores + self.get_scores()  # adds up word scores for each beam

        # update score vector to the given new scores
        scores, score_indices = scores.view(-1).topk(len(self.beam))
        # (beam size,), (beam size,) / top beam size scores among all beams * k candidates
        new_beam = []
        for i, score_idx in enumerate(score_indices):
            score_idx = score_idx.item()
            r = score_idx // k
            w = words.view(-1)[score_idx].item()  # w: word index according to one of top k indices

            new_node = self.copy_node(beam[r])  # create new beam node

            if w == self.endtok:
                new_node.score = scores[i]
                self.finished_nodes.append(new_node)  # if a beam is over, append it to finished_nodes
            else:
                if new_node.just_started:  # if sentence starts
                    new_node.just_started = False
                    new_node.init_words.append(w)
                if w >= self.vocab_size:  # if word in [<method_0>, ...]
                    new_node.prevent = w
                if w == self.eostok:  # if sentence ends
                    new_node.just_started = True
                new_node.words.append(w)
                new_node.score = scores[i]
                new_node.h = hs[r, :].unsqueeze(0)
                new_node.c = cs[r, :].unsqueeze(0)
                new_node.context = context[r, :].unsqueeze(0)
                new_beam.append(new_node)

        self.beam = new_beam
        return len(self.beam) > 0  # return false if all beams ended
