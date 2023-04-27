"""
The majority of this code is the kfold.py script from from github project: 
https://github.com/ljvmiranda921/ud-tagalog-spacy

This has been updated to output all of the metrics for each span category for each fold to ./metrics/kfold.jsonl (once all folds are completed), not just the average of the top-level precision/recall/f-score. This is useful if wanting to assess performance on a per-fold, per-category basis.

Also added further information at the end of each fold to output the appended metrics and add a timestamp. 

Also added wandb (weights & biases) integration, although this only outputs the final scores for each fold and does not do the dynamic tracking while each fold is training. However, this is not really required and is still useful for running experiments / hyperparamter tuning.

"""


import random
import tempfile
from pathlib import Path
from typing import List, Optional
import scripts.functions
import datetime
import wandb
import os

import spacy
import srsly
import typer
from spacy.cli._util import parse_config_overrides, setup_gpu
from spacy.cli._util import show_validation_error
from spacy.tokens import DocBin
from spacy.training.corpus import Corpus
from spacy.training.initialize import init_nlp
from spacy.training.loop import train as train_nlp
from spacy.util import load_config
from wasabi import msg

def chunk(l: List, n: int):
    """Split a list l into n chunks of fairly equal number of elements"""
    k, m = divmod(len(l), n)
    return (l[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def get_all_except(l: List, idx: int):
    """Get all elements of a list except a given index"""
    return l[:idx] + l[(idx + 1) :]


def flatten(l: List) -> List:
    """Flatten a list of lists"""
    return [item for sublist in l for item in sublist]


app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main(
    # fmt: off
    ctx: typer.Context,  # this is only used to read additional arguments
    corpus_path: Path = typer.Argument(..., help="Path to the full corpus."),
    config_path: Path = typer.Argument(..., help="Path to the spaCy configuration file."),
    output_path: Path = typer.Option(..., "--output-path", "--output", "-o", help="Path to save the output scores (JSON)."),
    n_folds: int = typer.Option(10, "--n-folds", "-n", help="Number of folds for cross-validation.", show_default=True),
    lang: Optional[str] = typer.Option("tl", "--lang", "-l", help="Language vocab to use.", show_default=True),
    shuffle: bool = typer.Option(False, "--shuffle", "-f", help="Flag for shuffling data"),
    use_gpu: int = typer.Option(0, help="GPU id to use. Pass -1 to use the CPU."),
    # fmt: on
):
    """Train a dependency parser with k-fold cross validation

    This command-line interface allows training a spaCy pipeline using k-fold
    cross validation. You can set the number of folds by passing a parameter to
    '--n-folds'. It performs the split automatically, so you need to pass the
    full corpus (not split into training/dev) in 'corpus_path'. Lastly,
    we get the average of the scores for each fold to obtain the final metrics.
    """
    if n_folds <= 1:
        raise ValueError("Cannot have folds less than or equal to 1.")

    overrides = parse_config_overrides(ctx.args)
    setup_gpu(use_gpu)

    nlp = spacy.blank(lang)
    doc_bin = DocBin().from_disk(corpus_path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    if shuffle:
        msg.info("Shuffling docs")
        random.shuffle(docs)

    folds = list(chunk(docs, n_folds))
    #Initialise an empty list (intially was an empty dict with specfic fields all_scores = {metric: [] for metric in METRICS} )
    all_scores= []
    
    
    for idx, fold in enumerate(folds):
        
        
        dev = fold
        train = flatten(get_all_except(folds, idx=idx))
        msg.divider(f"Fold {idx+1}, train: {len(train)}, dev: {len(dev)}")

        # Save the train and test dataset into a temporary directory
        # then train within the context of that directory
        ### Change the temporary files to be on the /datadrive folder 
        with tempfile.TemporaryDirectory(dir = os.environ.get('TMPDIR')) as tmpdir:
            
            msg.info("Splitting data for training")
            overrides["paths.train"] = str(Path(tmpdir)/"tmp_train.spacy")
            overrides["paths.dev"] = str(Path(tmpdir)/"tmp_dev.spacy")
            tmp_train_docbin = DocBin(docs=train)
            tmp_train_docbin.to_disk(overrides["paths.train"])
            tmp_dev_docbin = DocBin(docs=dev)
            tmp_dev_docbin.to_disk(overrides["paths.dev"])
            msg.good(
                f"Temp files at {overrides['paths.train']} and {overrides['paths.dev']}"
            )
                    
            msg.info("Validating configuration")
            with show_validation_error(config_path, hint_fill=False):
                config = load_config(config_path, overrides, interpolate=False)
                nlp = init_nlp(config)
           
            ##Initialise the wandb run, saving the spacy config
            wandb.init(project="wandb_MRI_NLP", config=config)
            
            #Train the model on the current fold
            msg.info("Training model for the current fold")
            nlp, _ = train_nlp(nlp, None)
                
            corpus = Corpus(overrides["paths.dev"], gold_preproc=False)
            dev_dataset = list(corpus(nlp))
            msg.info(f"Evaluating on the dev dataset...")
            scores = nlp.evaluate(dev_dataset)
            msg.info("Appending metric scores:")
            for metric in scores:
                msg.info(f"{metric} score of {scores[metric]}")
            all_scores.append(scores)
            now_time = str(datetime.datetime.now())
            msg.good(f"Finished evaluating fold {idx+1} at {now_time}")
            msg.good("Sending scores to wandb and finishing process")
            wandb.log(scores)
            wandb.finish()

    if output_path is not None:
        srsly.write_json(output_path, all_scores)
        msg.good(f"Saved results to {output_path}")

if __name__ == "__main__":
    app()