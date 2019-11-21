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

from tqdm import tqdm



def update_lr(o, args, epoch):
    if epoch % args.lrstep == 0:
        o.param_groups[0]['lr'] = args.lrhigh
    else:
        o.param_groups[0]['lr'] -= args.lr_delta


def train(model, optimizer, dataset, args):
    """
    input words => indices all removed from tags / trained to output indexed tags
    target words => indices included
    """
    model.train()

    print("Training", end=" ")
    loss = 0
    example_num = 0
    train_order = [('t1', dataset.t1_iter), ('t2', dataset.t2_iter), ('t3', dataset.t3_iter)]
    shuffle(train_order)

    for idx, train_iter in train_order:
        print(f"Training on {idx}")
        for count, batch in enumerate(train_iter):
            if count % 100 == 99:
                print(f"Example {example_num} / Average Loss: {loss / example_num}")
            batch = dataset.collate_fn(batch)
            out, _ = model(batch)  # out: (batch_size, max abstract len, target vocab size + max entity num)
            out = out[:, :-1, :].contiguous()  # exclude last words from each abstract

            tgt = batch.tgt[:, 1:].contiguous().view(-1).to(args.device)  # exclude first word from each target
            batch_loss = F.nll_loss(out.contiguous().view(-1, out.size(2)), tgt, ignore_index=1)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            loss += batch_loss.item() * len(batch.tgt)
            optimizer.step()
            optimizer.zero_grad()

            example_num += len(batch.tgt)
    loss = loss / example_num

    print("Average Train Loss: ", loss, end="\t")
    if loss < 100:
        print("PPL: ", exp(loss))


def evaluate(model, dataset, args):
    print("Evaluating", end="\t")
    model.eval()
    loss = 0
    example_num = 0

    iterator = dataset.make_test(args) if args.eval else dataset.val_iter
    for batch in tqdm(iterator):
        batch = dataset.collate_fn(batch)
        out, _ = model(batch)
        out = out[:, :-1, :]

        if example_num == 0:
            gen = out[0].max(1)[1]
            print(dataset.create_sentence(gen, batch.rawent))

        tgt = batch.tgt[:, 1:].contiguous().view(-1).to(args.device)
        batch_loss = F.nll_loss(out.contiguous().view(-1, out.size(2)), tgt, ignore_index=1)
        loss += batch_loss.item() * len(batch.tgt)
        example_num += len(batch.tgt)

    loss = loss / example_num
    print("VAL LOSS: ", loss, end="\t")
    if loss < 100:
        print(" PPL: ", exp(loss))
    model.train()
    return loss


def main(args):
    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    dataset = PaperDataset(args)
    args = set_args(args, dataset)
    model = Model(args)

    print(f"Device: {args.device}")
    model = model.to(args.device)
    if args.ckpt:
        cpt = torch.load(args.ckpt)
        model.load_state_dict(cpt)
        start_epoch = int(args.ckpt.split("/")[-1].split(".")[0]) + 1
        args.lr = float(args.ckpt.split("-")[-1])

        print(f"Model parameter restored to {args.ckpt}")
    else:
        with open(args.save + "/commandLineArgs.txt", 'w') as f:
            f.write("\n".join(sys.argv[1:]))
        start_epoch = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # early stopping based on Val Loss
    latest_loss = 1000000

    for e in range(start_epoch, args.epochs):
        print("Epoch: ", e, "\tLearning Rate: ", optimizer.param_groups[0]['lr'])
        train(model, optimizer, dataset, args)
        valid_loss = evaluate(model, dataset, args)
        if args.lrwarm:
            update_lr(optimizer, args, e)

        print("Saving model")
        torch.save(model.state_dict(),
                   args.save + "/" + str(e) + ".vloss-" + str(valid_loss)[:8] + ".lr-" + str(optimizer.param_groups[0]['lr']))

        if valid_loss > latest_loss:
            if args.lrdecay:
                print("Learning rate decayed.")
                optimizer.param_groups[0]['lr'] *= 0.5
        latest_loss = valid_loss


if __name__ == "__main__":
    args = parse_args()
    main(args)
