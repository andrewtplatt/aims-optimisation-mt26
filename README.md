# AIMS Optimisation - MT26

Code and exercises for the **Optimisation** course at the University of Oxford's AIMS CDT, Michaelmas Term 2026.

## Setup

Dependencies are managed using [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Extract the MNIST datasets:

```bash
unzip mnist_train.csv.zip
unzip mnist_test.csv.zip
```

## Contents

- `sudoku.py` - Sudoku solution as optimisation.
- `mnist.py` - MNIST SVM example.
- `projection.py` - Projection methods.
- `signal_recovery.py` - Signal recovery.

## Running

Run an example using:

```bash
uv run python sudoku.py
```