import argparse
import sacrebleu


parser = argparse.ArgumentParser(description="Compute BLEU with sacrebleu")
parser.add_argument("--preds", required=True, help="File with system outputs (one per line)")
parser.add_argument("--refs", required=True, help="File with references (one per line)")
args = parser.parse_args()


with open(args.preds, encoding="utf-8") as f:
preds = [ln.rstrip("\n") for ln in f]
with open(args.refs, encoding="utf-8") as f:
refs = [ln.rstrip("\n") for ln in f]


bleu = sacrebleu.corpus_bleu(preds, [refs])
print(f"BLEU = {bleu.score:.2f}")
