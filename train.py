from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import torch
from transformers import GPT2Config, GPT2LMHeadModel

tokenizer = PreTrainedTokenizerFast(tokenizer_file="model/tokenizer.json")

# update this to match your hardware
# multi gpu training is not really tested yet, i had some issues with it
N_CPU = 16
N_GPU = 1

model_config = GPT2Config(
    vocab_size=len(tokenizer),
    activation_function="silu",
    n_positions=512,
    n_ctx=512,
    n_embd=1024,
    n_head=16,
    n_layer=18,
)

# build the model
model = GPT2LMHeadModel(model_config)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

BATCH_SIZE = 128


dataset = load_dataset("nsarrazin/lichess-games-2023-01", split="train")

# only consider games that end in checkmate
# the assumption is that these games are higher quality
dataset = dataset.filter(lambda x: x["checkmate"] == True, num_proc=N_CPU)

print(f"Number of games: {len(dataset)}")

# add the special tokens for win/lose detection
# these can be used for doing cool things like beam search
# to improve performance 
def add_special_tokens(example):
    batch = []
    for score, moves in zip(example["result"], example["moves"]):
        if score == 0b10:
            moves = moves + ["<WHITE_WIN>"]
        elif score == 0b01:
            moves = moves + ["<BLACK_WIN>"]
        elif score == 0b11 or score == 0b00:
            moves = moves + ["<DRAW>"]
        else:
            raise ValueError(f"Unknown score: {score}")
        batch.append(" ".join(moves))
    return {"moves": batch}


dataset = dataset.map(add_special_tokens, batched=True, num_proc=N_CPU)

# tokenize the moves
dataset = dataset.map(
    lambda batch: tokenizer(
        batch["moves"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    ),
    batched=True,
    num_proc=N_CPU,
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

dataset = dataset.shuffle(seed=42)
# Split the dataset into train and test
dataset = dataset.train_test_split(test_size=1e-4)


# cosine annealing with warmup

training_args = TrainingArguments(
    per_device_train_batch_size=128,
    learning_rate=1e-4,
    weight_decay=0.01,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    bf16=True,
    num_train_epochs=1,
    optim="adamw_torch",
    report_to="wandb", # i use wandb for logging, you will need to login for it iirc
    eval_steps=1000,
    save_steps=5000,
    logging_steps=10,
    eval_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    hub_strategy="checkpoint",
    resume_from_checkpoint="last-checkpoint",
    push_to_hub=True,
    hub_model_id="nsarrazin/chessformer", # change this to your username/model_name
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    output_dir="model/chessformer",
    save_safetensors=False,
)

# cosine annealing with warmup, doing a single half cycle
# i think a lot could be done here to make this better
# for my use case that was good enough, i wanted to go for a 20:1 token/parameter ratio
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=10000,
    num_training_steps=len(dataset["train"])
    // (
        N_GPU
        * training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    ),
    num_cycles=0.5,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler),
)

try:
    trainer.train(resume_from_checkpoint=True)
except KeyboardInterrupt:
    print("Keyboard interrupt detected. Saving model before exiting...")
    trainer.save_model()
    trainer.push_to_hub(message="Training interrupted. Saving model before exiting...")
    raise
