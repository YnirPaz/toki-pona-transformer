# Toki Pona Transformer

A small decoder-only transformer for Toki Pona text generation implemented with pytorch. This is mostly a learning project/experiment to see how well a transformer with very little training data can perform. It seemed to have learned toki pona's syntax fairly well, but not much semantics.

The model has four causal attention layers, a context length of 256 tokens, and a 235-token vocabulary. The tokenizer gives each of 137 common Toki Pona words (nimi ku suli) one token, spells unfamiliar words letter by letter, and represents punctuation and newlines separately. `small` has 493,824 parameters; `medium` and its dropout variants have 855,040. See [`model.py`](model.py), [`tokenizer.py`](tokenizer.py), and [`training.py`](training.py) for the implementations.

## Prepare the data

Use Python 3.11 or later. Install the dependencies and run one command from the repository root:

```bash
python -m pip install -r requirements.txt
python download_dataset.py
python download_dataset.py --check
```

The preparation command downloads pinned source versions into `data/raw/`, regenerates the selected documents, verifies every text file against the checked-in hashes in `dataset_locks/`, and writes token arrays. The first run takes several minutes and needs roughly 100 MB for the downloads plus space for about 64,000 small text files. Later runs verify the existing output; use `--rebuild` to build it again. `data/` and trained model files are ignored by Git.

Each corpus has the same layout:

```text
data/processed/
  tokenizer.json
  kulupu-lapo/          # also tatoeba/ and wikipesija/
    manifest.json       # source ID, split, text hash, token offset and count
    metadata.jsonl      # one provenance record per document
    training/*.txt      # also validation/ and evaluation/
    training.uint16le   # also validation.uint16le and evaluation.uint16le
```

Each `.uint16le` file contains document token IDs concatenated in manifest order. Offsets and counts in `manifest.json` locate individual documents. The trainer reads the `.txt` files and tokenizes them again while shuffling document order each epoch; the token arrays are a checked, reusable record of exactly how each document tokenizes. Both use the same `TokiPonaTokenizer` implementation.

The selection files in `dataset_locks/` preserve the exact source IDs and training/validation/evaluation assignments from the original local datasets. This includes earlier manual quality exclusions, so the prepared **text bytes and document splits** match those datasets. Raw downloads and generated data are deliberately kept out of the repository. If a source or cleaning rule changes, preparation fails at the affected document instead of silently producing a different experiment.

| Corpus | Retained documents (train / validation / evaluation) | Source and rights |
| --- | ---: | --- |
| kulupu Lapo | 1,353 / 169 / 169 | [poki](https://github.com/kulupu-lapo/poki), individual documents have their own author and license metadata; some do not grant redistribution |
| Tatoeba | 47,501 / 5,945 / 5,938 | [NetherQuartz/tatoeba-tokipona](https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/); source sentence IDs and links are retained |
| Wikipesija | 2,404 / 301 / 301 | [NetherQuartz/wikipesija](https://huggingface.co/datasets/NetherQuartz/wikipesija), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); article IDs, titles, and URLs are retained |

The original preparation cleaned Markdown and screened unusable kulupu Lapo documents, deduplicated Tatoeba sentences by case-folded word sequence and against longer kulupu Lapo texts, and deduplicated Wikipesija articles against the long-form corpus. Tatoeba sentences found inside Wikipesija articles were excluded. A tokenization quality audit excluded one more Tatoeba sentence and 53 Wikipesija articles. Splits were made by whole document, roughly 80/10/10; kulupu Lapo and Wikipesija were balanced by token count, and Tatoeba was assigned with a fixed random seed. The older scripts in `scripts/` show the original cleaning and selection algorithms. The lock files capture their final output, including the quality decisions, in a compact reproducible form.

The source revisions are pinned in [`scripts/prepare_datasets.py`](scripts/prepare_datasets.py). kulupu Lapo is pinned to `275f55ef1c349880bf48c769e7241d4c6ed7fd6e`; the two Hugging Face datasets are pinned to `0b2b008f4da75cf02c9bf4b0b4f2aaa292689ca2` and `19dd7d1abf276e8d083f073d9e6700d536f38128`. The script checks the downloaded file hashes too. kulupu Lapo source documents are read directly from its archive so long source filenames work on Windows.

## Train and inspect

```bash
python main.py --model small --epochs 50
python main.py --model medium-do0.1 --epochs 50
python autoreg_generation_ui.py --run runs/small --checkpoint best
python scripts/evaluate_models.py --runs runs/small --dataset data/processed/kulupu-lapo
```

`main.py` now trains on the three prepared corpora by default. Training output goes under `runs/<model>/`; it includes the tokenizer JSON, history, and checkpoints. Resume with `python main.py --model small --resume runs/small/checkpoint.pt --resume-epochs 5`. The Tk interface in `autoreg_generation_ui.py` samples a saved checkpoint and shows token probabilities. Use the evaluation script to measure a checkpoint on a chosen corpus's evaluation split.

For the earlier local experiments, with a smaller dataset, the 855k-parameter model reached 2.424 without dropout (before starting to overfit) and 2.367 with 0.1 dropout. A later run with a larger corpus reached validation loss 1.999.

## Limits

The corpora are small and have different writing styles. Some content overlaps may remain despite word-sequence matching, and the 256-token context is short for longer documents. The saved split assignments reproduce the original corpus rather than resampling new upstream material. Each `runs/` folder stores its own tokenizer, but old checkpoints do not record a full dataset fingerprint. Keep that in mind when comparing old runs with a newly prepared corpus.

## License

The project code is released under the [WTFPL, version 2](LICENSE). This license does not cover the third-party datasets; their terms are listed above and in their source repositories.
