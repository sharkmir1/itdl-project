import sys
from random import shuffle
import os
from math import exp
import torch
from torch import nn
from torch.nn import functional as F
from dataset import PaperDataset
from args import parse_args, set_args
from models.model import Model


def update_lr(o, args, epoch):
    if epoch % args.lrstep == 0:
        o.param_groups[0]['lr'] = args.lrhigh
    else:
        o.param_groups[0]['lr'] -= args.lr_delta


def train(model, o, dataset, args):
    """
    input words => indices all removed from tags / trained to output indexed tags
    target words => indices included
    """
    model.train()

    print("Training", end=" ")
    loss = 0
    ex = 0
    train_order = [('t1', dataset.t1_iter), ('t2', dataset.t2_iter), ('t3', dataset.t3_iter)]
    shuffle(train_order)

    for idx, train_iter in train_order:
        print(f"Training on {idx}")
        for count, batch in enumerate(train_iter):
            if count % 100 == 99:
                print(ex, "of like 40k -- current avg loss ", (loss / ex))
            batch = dataset.collate_fn(batch)
            p, z, planlogits = model(batch)  # p: (batch_size, max abstract len, target vocab size + max entity num)
            p = p[:, :-1, :].contiguous()  # exclude last words from each abstract

            tgt = batch.tgt[:, 1:].contiguous().view(-1).to(args.device)  # exclude first word from each target
            loss = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt, ignore_index=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            loss += loss.item() * len(batch.tgt)
            o.step()
            o.zero_grad()
            ex += len(batch.tgt)
    loss = loss / ex
    print("AVG TRAIN LOSS: ", loss, end="\t")
    if loss < 100: print(" PPL: ", exp(loss))


def evaluate(m, ds, args):
    print("Evaluating", end="\t")
    m.eval()
    loss = 0
    ex = 0
    for b in ds.val_iter:
        b = ds.collate_fn(b)
        p, z, planlogits = m(b)
        p = p[:, :-1, :]
        tgt = b.tgt[:, 1:].contiguous().view(-1).to(args.device)
        l = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt, ignore_index=1)
        if ex == 0:
            g = p[0].max(1)[1]
            print(ds.reverse(g, b.rawent))
        loss += l.item() * len(b.tgt)
        ex += len(b.tgt)
    loss = loss / ex
    print("VAL LOSS: ", loss, end="\t")
    if loss < 100: print(" PPL: ", exp(loss))
    m.train()
    return loss


def main(args):
    try:
        os.stat(args.save)
        input("Save File Exists, OverWrite? <CTL-C> for no")
    except:
        os.mkdir(args.save)
    ds = PaperDataset(args)
    args = set_args(args, ds)
    m = Model(args)
    print(args.device)
    m = m.to(args.device)
    if args.ckpt:
        cpt = torch.load(args.ckpt)
        m.load_state_dict(cpt)
        starte = int(args.ckpt.split("/")[-1].split(".")[0]) + 1
        args.lr = float(args.ckpt.split("-")[-1])
        print('ckpt restored')

    else:
        with open(args.save + "/commandLineArgs.txt", 'w') as f:
            f.write("\n".join(sys.argv[1:]))
        starte = 0
    o = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)

    # early stopping based on Val Loss
    lastloss = 1000000

    for e in range(starte, args.epochs):
        print("epoch ", e, "lr", o.param_groups[0]['lr'])
        train(m, o, ds, args)
        vloss = evaluate(m, ds, args)
        if args.lrwarm:
            update_lr(o, args, e)
        print("Saving model")
        torch.save(m.state_dict(),
                   args.save + "/" + str(e) + ".vloss-" + str(vloss)[:8] + ".lr-" + str(o.param_groups[0]['lr']))
        if vloss > lastloss:
            if args.lrdecay:
                print("decay lr")
                o.param_groups[0]['lr'] *= 0.5
        lastloss = vloss


if __name__ == "__main__":
    args = parse_args()
    main(args)
