# Reproduction of the Chessformer model

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-nsarrazin/chessformer-yellow?logo=huggingface)](https://huggingface.co/nsarrazin/chessformer)

This is a collection of scripts that I used to train my [chessformer](https://huggingface.co/nsarrazin/chessformer) model.

It's a 231M parameter model, trained on 4.4B tokens of lichess games. The games are represented as UCI moves, with each token being a move.

The goal was not to achieve very good ELO performance, but to have a model based on a well known architecture for LLMs that can then be analyzed for interpretability. See this notebook for more info.

> [!WARNING]
> Like all weekend projects, I didn't really document along, so I just collected my scripts after the fact. Things might or might not work, I'm happy to answer questions and provide some support on a best effort basis though!


### Training run

The model was trained on a single H100 for about 48 hours over the course of a few days.

| Eval Loss | Train Loss | Learning Rate |
|:---:|:---:|:---:|
| ![Eval Loss](docs/assets/eval_loss.png) | ![Train Loss](docs/assets/train_loss.png) | ![Learning Rate](docs/assets/lr.png) |


The loss has some really big upward spikes. If I had to guess it's because I represent each UCI move as its own token, including pawn promotion moves. 

Each possible promotion move has its own token so for example `b2a1q` is a pawn promotion move but it's unlikely that the model will see them all in the training set. When the model sees an untrained promotion move, the loss will spike.

I also messed up my learning rate scheduling initially, had to stop the run to fix it. Don't think it impacted the model too much but something to do better next time.

You can find the training script [here](train.py) and the script used for creating the tokenizer [here](tokenizer.py).

### What I would like to do next

- Try curriculum learning: At first train on all kind of games and then bias towards high ELO games as the learning rate decreases.

- Improve tokenization by adding tokens for promotion and deprecating the uci notation so that `b2a1q` would become `b2a1 <PROMOTE_QUEEN>`. That way the model would only have to train on 4 kind of promotion moves (q, r, b, k)

- Fine tune the modle using reinforcement learning. using stockfish centipawns as a reward function.

- Train sparse autoencoders on the model's layers to see what kind of features the model learns to represent.