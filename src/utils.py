import pytest


def add_special_tokens(example):
    """
    Function extracted from train.py for testing.
    Processes chess moves to add promotion tokens and game outcome tokens.
    Now processes a single example instead of a batch.
    """
    score = example["result"]
    moves = example["moves"]

    # replace the promotion moves with a standard move + promotion token
    # ["a1a2q"] becomes ["a1a2", "<PROMOTE_QUEEN>"]
    expanded_moves = []
    promotion_map = {
        "q": "<PROMOTE_QUEEN>",
        "r": "<PROMOTE_ROOK>",
        "b": "<PROMOTE_BISHOP>",
        "n": "<PROMOTE_KNIGHT>",
    }

    for move in moves:
        if len(move) == 5 and move[4] in promotion_map:
            # This is a promotion move, split it
            expanded_moves.append(move[:4])
            expanded_moves.append(promotion_map[move[4]])
        else:
            expanded_moves.append(move)
    moves = expanded_moves

    # add the end of game tokens
    if score == 0b10:
        moves = moves + ["<WHITE_WIN>"]
    elif score == 0b01:
        moves = moves + ["<BLACK_WIN>"]
    elif score == 0b11 or score == 0b00:
        moves = moves + ["<DRAW>"]
    else:
        raise ValueError(f"Unknown score: {score}")

    return {"moves": " ".join(moves)}


def test_white_wins():
    example = {"result": 0b10, "moves": ["e2e4", "e7e5", "f2f4"]}
    result = add_special_tokens(example)
    assert result["moves"] == "e2e4 e7e5 f2f4 <WHITE_WIN>"


def test_black_wins():
    example = {"result": 0b01, "moves": ["e2e4", "e7e5", "f2f4"]}
    result = add_special_tokens(example)
    assert result["moves"] == "e2e4 e7e5 f2f4 <BLACK_WIN>"


def test_draw():
    example = {"result": 0b11, "moves": ["e2e4", "e7e5"]}
    result = add_special_tokens(example)
    assert result["moves"] == "e2e4 e7e5 <DRAW>"


def test_queen_promotion():
    example = {"result": 0b10, "moves": ["e2e4", "e7e8q"]}
    result = add_special_tokens(example)
    assert result["moves"] == "e2e4 e7e8 <PROMOTE_QUEEN> <WHITE_WIN>"


def test_rook_promotion():
    example = {"result": 0b01, "moves": ["e2e4", "e7e8r"]}
    result = add_special_tokens(example)
    assert result["moves"] == "e2e4 e7e8 <PROMOTE_ROOK> <BLACK_WIN>"


def test_bishop_promotion():
    example = {"result": 0b11, "moves": ["e2e4", "e7e8b"]}
    result = add_special_tokens(example)
    assert result["moves"] == "e2e4 e7e8 <PROMOTE_BISHOP> <DRAW>"


def test_knight_promotion():
    example = {"result": 0b00, "moves": ["e2e4", "e7e8n"]}
    result = add_special_tokens(example)
    assert result["moves"] == "e2e4 e7e8 <PROMOTE_KNIGHT> <DRAW>"


def test_invalid_score():
    example = {"result": 5, "moves": ["e2e4"]}
    with pytest.raises(ValueError):
        add_special_tokens(example)


if __name__ == "__main__":
    pytest.main([__file__])
