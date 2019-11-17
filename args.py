import torch
import argparse


def set_args(args, ds):
    args.output_vocab_size = len(ds.OUTPUT.vocab)
    args.target_vocab_size = len(ds.TARGET.vocab)
    args.input_vocab_size = len(ds.INPUT.vocab)
    args.entity_vocab_size = len(ds.ENT.itos)
    args.relation_vocab_size = len(ds.REL.itos)
    args.start_token = ds.OUTPUT.vocab.stoi["<start>"]
    args.ent_vocab = ds.ENT.itos
    args.inp_vocab = ds.INPUT.vocab.itos
    args.lr_delta = (args.lrhigh - args.lr) / args.lrstep
    args.esz = args.hsz  # embedding size == hidden size
    return args


def parse_args():
    parser = argparse.ArgumentParser(description='Abstract Generation')

    # model
    parser.add_argument("-model", default="graph")
    parser.add_argument("-esz", default=500, type=int)
    parser.add_argument("-hsz", default=500, type=int)
    parser.add_argument("-layers", default=2, type=int)
    parser.add_argument("-drop", default=0.1, type=float)
    parser.add_argument("-transformer_drop", default=0.1, type=float)
    parser.add_argument("-embdrop", default=0, type=float)
    parser.add_argument("-gdrop", default=0.3, type=float)
    parser.add_argument("-transformer_num", default=6, type=int)
    parser.add_argument("-attnheads", default=3, type=int)
    parser.add_argument("-notitle", action='store_true')
    parser.add_argument("-elmoVocab", default="../data/elmoVocab.txt", type=str)
    parser.add_argument("-elmo", action='store_true')
    parser.add_argument("-heads", default=4, type=int)

    # training and loss
    parser.add_argument("-cl", default=None, type=float)
    parser.add_argument("-bsz", default=32, type=int)
    parser.add_argument("-vbsz", default=32, type=int)
    parser.add_argument("-epochs", default=20, type=int)
    parser.add_argument("-clip", default=1, type=float)

    # optimization
    parser.add_argument("-lr", default=0.1, type=float)
    parser.add_argument("-lrhigh", default=0.5, type=float)
    parser.add_argument("-lrstep", default=4, type=int)
    parser.add_argument("-lrwarm", action="store_true")
    parser.add_argument("-lrdecay", action="store_true")
    parser.add_argument('-max_grad_norm', type=int, default=1)

    # data
    parser.add_argument("-nosave", action='store_false')
    parser.add_argument("-save", required=True)
    parser.add_argument("-outunk", default=5, type=int)
    parser.add_argument("-entunk", default=5, type=int)
    parser.add_argument("-datadir", default="data/")
    parser.add_argument("-data", default="preprocessed.train.tsv")
    parser.add_argument("-relvocab", default="relations.vocab", type=str)
    parser.add_argument("-savevocab", default=None, type=str)
    parser.add_argument("-loadvocab", default=None, type=str)
    # eval
    parser.add_argument("-eval", action='store_true')

    # inference
    parser.add_argument("-maxlen", default=200, type=int)
    parser.add_argument("-test", action='store_true')
    parser.add_argument("-sample", action='store_true')
    parser.add_argument("-inputs", default="../data/fullGraph.test.tsv", type=str)

    parser.add_argument("-t1size", default=32, type=int)
    parser.add_argument("-t2size", default=24, type=int)
    parser.add_argument("-t3size", default=8, type=int)
    parser.add_argument("-title", action='store_true')
    parser.add_argument("-ckpt", default=None, type=str)
    parser.add_argument("-plweight", default=0.2, type=float)
    parser.add_argument("-entdetach", action='store_true')

    args = parser.parse_args()
    args.gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(args.gpu)
    # torch.cuda.set_device(args.gpu)

    return args
