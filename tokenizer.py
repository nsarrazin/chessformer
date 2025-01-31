from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# generate the list of all valid chess UCI moves
uci_moves = []

files = ["a", "b", "c", "d", "e", "f", "g", "h"]
ranks = [str(i) for i in range(1, 9)]

cells = [file + rank for file in files for rank in ranks]

uci_moves = [start_cell + end_cell for start_cell in cells for end_cell in cells]

# add promotion moves
# which can only occur when a pawn reaches the opposite end of the board
# this is actually super inefficient since pawns can only move forward or diagonally on capture

# so we have a lot of redundant tokens
# we could also just have a promote token, because a lot of the tokens here are undertrained
# and cause the model to error
pieces = ["q", "r", "b", "n"]
white_promotion_start_cells = [file + "7" for file in files]  # white pawns on the 7th rank
white_promotion_end_cells = [file + "8" for file in files]    # white pawns moving to the 8th rank
black_promotion_start_cells = [file + "2" for file in files]  # black pawns on the 2nd rank
black_promotion_end_cells = [file + "1" for file in files]    # black pawns moving to the 1st rank

promotion_moves = [start + end + piece 
                   for start in white_promotion_start_cells + black_promotion_start_cells
                   for end in white_promotion_end_cells + black_promotion_end_cells
                   for piece in pieces
                   if (start in white_promotion_start_cells and end in white_promotion_end_cells) or
                      (start in black_promotion_start_cells and end in black_promotion_end_cells)]

uci_moves.extend(promotion_moves)

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
trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "<WHITE_WIN>", "<BLACK_WIN>", "<DRAW>"])
tokenizer.train(["vocab.txt"], trainer)

# Save the tokenizer
tokenizer.save("model/tokenizer.json")

print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
