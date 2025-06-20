from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# generate the list of all valid chess UCI moves
uci_moves = []

files = ["a", "b", "c", "d", "e", "f", "g", "h"]
ranks = [str(i) for i in range(1, 9)]

cells = [file + rank for file in files for rank in ranks]

uci_moves = [start_cell + end_cell for start_cell in cells for end_cell in cells]

# write vocab.txt
with open("vocab.txt", "w") as vocab_file:
    for move in uci_moves:
        vocab_file.write(move + "\n")

# Create a new tokenizer
tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

# Set the pre-tokenizer to split on whitespace
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Train the tokenizer on our vocab file
trainer = trainers.WordLevelTrainer(
    special_tokens=[
        "[UNK]",
        "[PAD]",
        "<WHITE_WIN>",
        "<BLACK_WIN>",
        "<DRAW>",
        "<PROMOTE_QUEEN>",
        "<PROMOTE_ROOK>",
        "<PROMOTE_BISHOP>",
        "<PROMOTE_KNIGHT>",
    ]
)
tokenizer.train(["vocab.txt"], trainer)

# Save the tokenizer
tokenizer.save("model/tokenizer.json")

print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
