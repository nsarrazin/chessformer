from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    get_constant_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
import chess.pgn
import io
from datasets import load_dataset
import torch
from transformers import LlamaForCausalLM, LlamaConfig
from utils import add_special_tokens

# update this to match your hardware
# multi gpu training is not really tested yet, i had some issues with it
N_CPU = 16
N_GPU = 1

tokenizer = PreTrainedTokenizerFast(tokenizer_file="model/tokenizer.json")


# Check if CUDA is available and handle device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Aborting.")
    exit(1)


def parse_movetext_to_uci(movetext):
    """Convert PGN movetext to list of UCI moves"""
    try:
        # Create a StringIO object from the movetext
        pgn_io = io.StringIO(movetext)
        game = chess.pgn.read_game(pgn_io)

        if game is None:
            return []

        board = game.board()
        uci_moves = []

        for move in game.mainline_moves():
            uci_moves.append(move.uci())
            board.push(move)

        return uci_moves
    except:
        return []


def map_result(x):
    if x["Result"] == "1-0":
        return 0b10
    elif x["Result"] == "0-1":
        return 0b01
    elif x["Result"] == "1/2-1/2":
        return 0b11
    else:
        return 0b00


ds = (
    load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    .filter(
        lambda x: x["Termination"] == "Normal"
        and x["WhiteElo"] > 1200
        and x["BlackElo"] > 1200
    )
    .map(
        lambda x: {
            "moves": parse_movetext_to_uci(x["movetext"]),
            "result": map_result(x),
            "white_elo": x["WhiteElo"],
            "black_elo": x["BlackElo"],
            "avg_elo": round((x["WhiteElo"] + x["BlackElo"]) / 2),
        },
        remove_columns=[
            "Event",
            "Site",
            "White",
            "Black",
            "Result",
            "UTCDate",
            "UTCTime",
            "WhiteElo",
            "BlackElo",
            "WhiteRatingDiff",
            "BlackRatingDiff",
            "ECO",
            "Opening",
            "TimeControl",
            "Termination",
            "movetext",
            "WhiteTitle",
            "BlackTitle",
        ],
    )
    .filter(lambda x: x["result"] != 0b00 and len(x["moves"]) > 3)
    .map(add_special_tokens)
    .map(
        lambda batch: tokenizer(
            batch["moves"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        ),
        batched=True,
    )
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

ds = ds.shuffle(seed=42).with_format("torch")

# Split dataset: first 10k for test, rest for train
test_ds = ds.take(10000)
train_ds = ds.skip(10000)

ds = {"train": train_ds, "test": test_ds}

model_config = LlamaConfig(
    vocab_size=len(tokenizer),
    hidden_act="silu",
    hidden_size=960,
    intermediate_size=2560,
    num_attention_heads=15,
    num_hidden_layers=32,
    num_key_value_heads=5,
    max_position_embeddings=512,
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=0,
    eos_token_id=0,
    initializer_range=0.02,
    rms_norm_eps=1e-05,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
)

# build the model
model = LlamaForCausalLM(model_config)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


training_args = TrainingArguments(
    per_device_train_batch_size=1024,
    learning_rate=1e-4,
    weight_decay=0.01,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    bf16=True,
    num_train_epochs=1,
    optim="adamw_torch",
    report_to="wandb",
    eval_steps=360,
    save_steps=360,
    logging_steps=4,
    eval_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    hub_strategy="checkpoint",
    resume_from_checkpoint="last-checkpoint",
    push_to_hub=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    hub_model_id="nsarrazin/chessformer-2",
    output_dir="model/chessformer-2",
    save_safetensors=False,
    max_steps=round(
        6e8
    ),  # kind of arbitrary because we use a streaming dataset and a fixed LR
    run_name="chessformer-2-prod",
    dataloader_num_workers=N_CPU - 1,
)
scheduler = get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps=200,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=ds["train"].with_format("torch"),
    eval_dataset=ds["test"].with_format("torch"),
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler),
)

try:
    trainer.train()
except KeyboardInterrupt:
    print("Keyboard interrupt detected. Saving model before exiting...")
    trainer.save_model()
    trainer.push_to_hub(message="Training interrupted. Saving model before exiting...")
    raise
