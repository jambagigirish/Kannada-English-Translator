import argparse
from models.translator import Translator


parser = argparse.ArgumentParser(description="Kannada → English translator")
parser.add_argument("--text", type=str, help="Input Kannada text")
parser.add_argument("--batch_file", type=str, default=None, help="Optional path to a file with one Kannada sentence per line")
args = parser.parse_args()


tr = Translator()


if args.batch_file:
with open(args.batch_file, "r", encoding="utf-8") as f:
lines = [ln.strip() for ln in f if ln.strip()]
outs = tr.translate_batch(lines)
for o in outs:
print(o)
else:
print(tr.translate(args.text))
