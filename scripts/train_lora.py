"""
AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model


parser = argparse.ArgumentParser()
parser.add_argument("--train_tsv", required=True)
parser.add_argument("--val_tsv", required=True)
parser.add_argument("--base_model", default=os.getenv("HF_BASE_MODEL", "Helsinki-NLP/opus-mt-kn-en"))
parser.add_argument("--output_dir", default="outputs/lora-kn-en")
parser.add_argument("--src_lang", default=os.getenv("SRC_LANG"))
parser.add_argument("--tgt_lang", default=os.getenv("TGT_LANG"))
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=3)
args = parser.parse_args()


# Load data from TSVs
train = load_dataset("csv", data_files=args.train_tsv, delimiter="\t", column_names=["src","tgt"])['train']
val = load_dataset("csv", data_files=args.val_tsv, delimiter="\t", column_names=["src","tgt"])['train']


tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)


# Optional language codes (for NLLB)
if hasattr(tokenizer, "src_lang") and args.src_lang:
tokenizer.src_lang = args.src_lang
if hasattr(tokenizer, "tgt_lang") and args.tgt_lang:
tokenizer.tgt_lang = args.tgt_lang


peft_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, peft_cfg)


def preprocess(examples):
model_inputs = tokenizer(examples["src"], max_length=args.max_length, truncation=True)
with tokenizer.as_target_tokenizer():
labels = tokenizer(examples["tgt"], max_length=args.max_length, truncation=True)
model_inputs["labels"] = labels["input_ids"]
return model_inputs


train_tok = train.map(preprocess, batched=True, remove_columns=train.column_names)
val_tok = val.map(preprocess, batched=True, remove_columns=val.column_names)


collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


training_args = Seq2SeqTrainingArguments(
output_dir=args.output_dir,
per_device_train_batch_size=args.batch_size,
per_device_eval_batch_size=args.batch_size,
learning_rate=5e-5,
num_train_epochs=args.epochs,
evaluation_strategy="epoch",
save_strategy="epoch",
predict_with_generate=True,
fp16=True
)


trainer = Seq2SeqTrainer(
model=model,
args=training_args,
train_dataset=train_tok,
eval_dataset=val_tok,
tokenizer=tokenizer,
data_collator=collator,
)


trainer.train()
trainer.save_model(args.output_dir)
