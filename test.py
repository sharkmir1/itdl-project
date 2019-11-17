import torch
from dataset import PaperDataset
from models.model import Model
from args import parse_args, set_args
from tqdm import tqdm


def test(args, dataset, model):
    model.eval()
    k = 0
    test_iter = dataset.make_test(args)
    result = open("../outputs/" + model + ".inputs.beam_predictions.cmdline", 'w')
    preds = []
    golds = []
    for batch in tqdm(test_iter):
        # if k == 10: break
        # print(k,len(data))
        batch = dataset.collate_fn(batch)
        gen = model.beam_generate(batch, beam_size=4, k=6)
        gen.sort()  # sort 'done' sequences by their scores
        gen = dataset.reverse(gen.done[0].words, batch.rawent)
        k += 1
        gold = dataset.reverse(batch.tgt[0][1:], batch.rawent)
        preds.append(gen.lower())
        golds.append(gold.lower())
        result.write(str(k) + '.\n')
        result.write("GOLD\n" + gold.encode('ascii', 'ignore').decode('utf-8', 'ignore') + '\n')
        result.write("GEN\n" + gen.encode('ascii', 'ignore').decode('utf-8', 'ignore').lower() + '\n\n')

    model.train()
    return preds, golds


if __name__ == "__main__":
    args = parse_args()
    args.eval = True
    ds = PaperDataset(args)
    args = set_args(args, ds)
    m = Model(args)
    cpt = torch.load(args.save, map_location=lambda storage, location: storage)
    m.load_state_dict(cpt)
    m = m.to(args.device)
    m.args = args
    m.maxlen = args.max
    m.starttok = ds.OUTPUT.vocab.stoi['<start>']
    m.endtok = ds.OUTPUT.vocab.stoi['<eos>']
    m.eostok = ds.OUTPUT.vocab.stoi['.']
    preds, gold = test(args, ds, m)
    '''
  scores = metrics(preds,gold)
  for k,v in scores.items():
    print(k+'\t'+str(scores[k]))
  '''
