import torch
from dataset import PaperDataset
from models.model import Model
from args import parse_args, set_args
from tqdm import tqdm

from train import evaluate


def test(args, dataset, model):
    model.eval()
    test_iter = dataset.make_test(args)
    result = open("./outputs/" + args.save.split("/")[-1] + ".inputs.beam_predictions.cmdline", 'w')
    preds = []
    golds = []
    for idx, batch in tqdm(enumerate(test_iter)):
        batch = dataset.collate_fn(batch)

        beam = model.beam_generate(batch, beam_size=4, k=6)
        beam.sort()  # sort finished_nodes sequences by their scores
        beam = dataset.create_sentence(beam.finished_nodes[0].words, batch.rawent)
        gold = dataset.create_sentence(batch.tgt[0][1:], batch.rawent)
        preds.append(beam.lower())
        golds.append(gold.lower())
        result.write(str(idx+1) + '.\n')
        result.write("GOLD\n" + gold.encode('ascii', 'ignore').decode('utf-8', 'ignore') + '\n')
        result.write("GEN\n" + beam.encode('ascii', 'ignore').decode('utf-8', 'ignore').lower() + '\n\n')

    return preds, golds


if __name__ == "__main__":
    args = parse_args()
    args.eval = True
    dataset = PaperDataset(args)
    args = set_args(args, dataset)
    model = Model(args)
    state_dict = torch.load(args.save, map_location=lambda storage, location: storage)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.args = args
    model.maxlen = args.maxlen
    model.starttok = dataset.OUTPUT.vocab.stoi['<start>']
    model.endtok = dataset.OUTPUT.vocab.stoi['<eos>']
    model.eostok = dataset.OUTPUT.vocab.stoi['.']
    # preds, gold = test(args, dataset, model)

    evaluate(model, dataset, args)

