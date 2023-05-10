"""
Calculation of precision and recall for spaCy spancat component (i.e. annotated vs predicted labels).
Includes confidence intervals, and can provide both overall performance and also on a per-label basis.

v1.0 - 27/04/23 - Initial attempt, heavily adapting code from prodigy forum (https://support.prodi.gy/t/show-false-negative-false-positives-in-ner/3223/4) to make work with spancat (rather than NER)
v1.1 - 30/04/23 - updated to keep list of spans internally (not just the length) so they can be assesed on a per label basis. Updated the function to allow passing a list of labels for assessment
v1.2 - 04/05/23 - Reworked to run as a script from the command line (rather than a .ipynb)
v1.3 - 10/05/23 - reworked to take a list of labels from the command line (actually provided as a variable within the project.yml), this needs to be passed a single list of comma separated labels without spaces, as typer cannot take an arbitrary number of variables.

labels = ['HVL', 'NO_HVL', 'RVL', 'NO_RVL', 'GVL', 'NO_GVL']

    # fmt: off
    ctx: typer.Context,  # this is only used to read additional arguments
    corpus_path: Path = typer.Argument(..., help="Path to the test/validation data."),
    config_path: Path = typer.Argument(..., help="Path to the spaCy configuration file."),
    model_path: Path = typer.Argument(..., help="Path to the spaCy model file."),
    per_label: bool = typer.Option(False, "--per-label", "-p", help="Flag for providing output on a per-label basis."),
    #labels: List = typer.Option([], "--confidence-labels", "-l", help="List of labels to assess, not required if per-label metrics not desired.", show_default=True),
    #labels: Annotated[Optional[List[str]], typer.Option("--labels")] = [],
    labels: str = typer.Option("", "--labels", "-p", help="Labels for assessment, provided as comma separated without spaces.")
    #files: List[Path], celebration: str):

    # fmt: on
    
This all needs to be changed to argparse as typer cannot take a list of strings as an input

"""

import spacy
from scipy.stats import beta
from spacy.lang.en import English
from spacy.tokens import Token, Span, Doc, DocBin
from spacy.training import Example
import pandas as pd
import numpy as np

from typing import List, Optional
from typing_extensions import Annotated
import typer
from pathlib import Path
from spacy.cli._util import parse_config_overrides
from spacy.util import load_config

def confusion_matrix(your_evaluation_data=None, ner_model = None, labels=None, per_label=False):
    #function that takes:
    # - your_evaluation_data, a list of spacy example objects
    # - ner_model, the loaded spacy model to be used to generate the predictions 
    # - labels, a list of labels to be used in assessment if per_label is Tru
    # - per_label, if true will output precision/recall with confifence intervals per label, in addition to the overall metrics
    
    #initialise empty lists for each cell of confusion matrix
    tp_list,fp_list,fn_list,tn_list = [],[],[],[]
    
    #generate list of text and example object pairs
    data_tuples = [(eg.text, eg) for eg in your_evaluation_data]
    
    #for each tuple, append to the confusion matrix lists the false positives, true positives, false negatives and true negatives
    for doc, example in ner_model.pipe(data_tuples, as_tuples=True):
        #generate list of the predicted spans
        predicted_spans = [(span.start, span.end, span.label_) for span in doc.spans['sc']]
        #generate lists of the manually annotated spans, from the example object
        correct_spans = [(span.start, span.end, span.label_) for span in example.reference.spans['sc']]
        # false positives
        for span in predicted_spans:
            if span not in correct_spans:
                fp_list.append(span)
        # true positives
        for span in predicted_spans:
            if span in correct_spans:
                tp_list.append(span)
        # false negatives
        for span in correct_spans:
            if span not in predicted_spans:
                fn_list.append(span)
        # true negatives
        for span in correct_spans:
            if span in predicted_spans:
                tn_list.append(span)
    
    #Generate confusion matrix for each label in turn
    if per_label == True:
        for label in labels:
            tp_label,fp_label,fn_label,tn_label = [],[],[],[]
            for span in tp_list:
                if span[2]==label: #span[2] here is the third item in the tp_list tuples, which is span.label_
                    tp_label.append(span)
            for span in fp_list:
                if span[2]==label:
                    fp_label.append(span)
            for span in fn_list:
                if span[2]==label:
                    fn_label.append(span)
            for span in tn_list:
                if span[2]==label:
                    tn_label.append(span)        
                     
            print(str(label) + ' label performance:')
            fp = len(fp_label)
            tp = len(tp_label)
            fn = len(fn_label)
            tn = len(tn_label)
            print("   True Positives: " + str(tp))
            print("   False Positives: " + str(fp))
            print("   False Negatives: " + str(fn))
            print("   True Negatives: " + str(tn))
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            print('   Precision = ' + str(np.round(precision, 2)) + ', 95% confidence interval ' + str((np.round((beta.ppf(0.025, tp+1, fp+1)),2), np.round((beta.ppf(0.975, tp+1, fp+1)), 2)))) # precision
            print('   Recall = ' + str(np.round(recall, 2)) + ', 95% confidence interval ' + str((np.round((beta.ppf(0.025, tp+1, fn+1)),2), np.round((beta.ppf(0.975, tp+1, fn+1)), 2)))) # precision
            
    #Overall performance is printed whether per_label is True or False
    print("Overall performance:")            
    fp = len(fp_list)
    tp = len(tp_list)
    fn = len(fn_list)
    tn = len(tn_list)
    print("   True Positives: " + str(tp))
    print("   False Positives: " + str(fp))
    print("   False Negatives: " + str(fn))
    print("   True Negatives: " + str(tn))
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print('   Precision = ' + str(np.round(precision, 2)) + ', 95% confidence interval ' + str((np.round((beta.ppf(0.025, tp+1, fp+1)),2), np.round((beta.ppf(0.975, tp+1, fp+1)), 2)))) # precision
    print('   Recall = ' + str(np.round(recall, 2)) + ', 95% confidence interval ' + str((np.round((beta.ppf(0.025, tp+1, fn+1)),2), np.round((beta.ppf(0.975, tp+1, fn+1)), 2)))) # precision


app = typer.Typer()

@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)    
    
def main(
    # fmt: off
    ctx: typer.Context,  # this is only used to read additional arguments
    corpus_path: Path = typer.Argument(..., help="Path to the test/validation data."),
    config_path: Path = typer.Argument(..., help="Path to the spaCy configuration file."),
    model_path: Path = typer.Argument(..., help="Path to the spaCy model file."),
    per_label: bool = typer.Option(False, "--per-label", "-p", help="Flag for providing output on a per-label basis."),
    #labels: List = typer.Option([], "--confidence-labels", "-l", help="List of labels to assess, not required if per-label metrics not desired.", show_default=True),
    #labels: Annotated[Optional[List[str]], typer.Option("--labels")] = [],
    labels: str = typer.Option("", "--labels", "-p", help="Labels for assessment, provided as comma separated without spaces.")
    #files: List[Path], celebration: str):

    # fmt: on
   ):   
   
    nlp = spacy.load(model_path)    
    doc_bin = DocBin().from_disk(corpus_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    examples = [Example(nlp(doc.text), doc) for doc in docs]
    label_list=labels.split(",")
    
    confusion_matrix(examples, nlp, label_list, per_label=True)

if __name__ == "__main__":
    app()