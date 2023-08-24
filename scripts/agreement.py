"""

TODO:
Adapt to work with the prodigy database access once have SSH access again

v1.0 31/07/23 - working version loading annotations from JSON, and both standard and relaxed matching
v1.1 23/08/23 - adapted to work as part of a spacy project

"""

import os
import json
import numpy as np
from scipy.stats import beta

#import pandas as pd

from typing import List, Optional
from typing_extensions import Annotated
import typer
from pathlib import Path
from spacy.cli._util import parse_config_overrides
from spacy.util import load_config

# Function to calculate precision, recall, and F1 score
def calculate_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def calculate_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calculate_f_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Function to calculate confidence interval using the beta distribution
def calculate_confidence_interval(tp, total, alpha=0.05):
    lower_bound = beta.ppf(alpha / 2, tp + 1, total - tp + 1)
    upper_bound = beta.ppf(1 - alpha / 2, tp + 1, total - tp + 1)
    return lower_bound, upper_bound

# Function to calculate precision, recall, and F1-score with confidence intervals
def calculate_precision_recall_f1_with_ci(matches, gold_total, model_total, alpha=0.05):
    tp = len(matches)
    fp = model_total - tp
    fn = gold_total - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    precision_ci = calculate_confidence_interval(tp, model_total, alpha)
    recall_ci = calculate_confidence_interval(tp, gold_total, alpha)
    f1_ci = calculate_confidence_interval(2 * tp, gold_total + model_total, alpha)

    return precision, recall, f1, precision_ci, recall_ci, f1_ci

# Function for getting the dictionaries of spans for each annotator
def get_dicts(annotation_dict, gold_annotator, model_annotator):
    gold_spans_dict = {}
    model_spans_dict = {}

    for input_hash, data in annotation_dict.items():
        gold_sessions = data.get(gold_annotator, {}).get('spans', [])
        model_sessions = data.get(model_annotator, {}).get('spans', [])

        gold_spans = set((session['start'], session['end'], session['label']) for session in gold_sessions)
        model_spans = set((session['start'], session['end'], session['label']) for session in model_sessions)

        gold_spans_dict[input_hash] = gold_spans
        model_spans_dict[input_hash] = model_spans

    return gold_spans_dict, model_spans_dict


def output_standard_matching(annotation_dict, gold_annotator, model_annotator):

    gold_spans_dict, model_spans_dict = get_dicts(annotation_dict, gold_annotator, model_annotator)

    # Calculate standard matches
    all_input_hashes = set(gold_spans_dict.keys()).union(model_spans_dict.keys())

    matches = []
    for input_hash in all_input_hashes:
        gold_spans = gold_spans_dict.get(input_hash, set())
        model_spans = model_spans_dict.get(input_hash, set())

        for model_span in model_spans:
            if model_span in gold_spans:
                matches.append(model_span)

    # Calculate precision, recall, and F1 with confidence intervals
    gold_total = sum(len(gold_spans) for gold_spans in gold_spans_dict.values())
    model_total = sum(len(model_spans) for model_spans in model_spans_dict.values())

    precision, recall, f1, precision_ci, recall_ci, f1_ci = calculate_precision_recall_f1_with_ci(matches, gold_total, model_total)

    print("Standard Matching:")
    print("Precision:", precision, ", 95% confidence interval:", precision_ci)
    print("Recall:", recall, ", 95% confidence interval:", recall_ci)
    print("F1 score:", f1, ", 95% confidence interval:", f1_ci)

    
def output_relaxed_matching(annotation_dict, gold_annotator, model_annotator):

    gold_spans_dict, model_spans_dict = get_dicts(annotation_dict, gold_annotator, model_annotator)
    
    # Calculate relaxed matches
    all_input_hashes = set(gold_spans_dict.keys()).union(model_spans_dict.keys())

    matches = []
    for input_hash in all_input_hashes:
        gold_spans = gold_spans_dict.get(input_hash, set())
        model_spans = model_spans_dict.get(input_hash, set())

        for model_span in model_spans:
            if any(model_span[0] <= gold_span[1] and model_span[1] >= gold_span[0] and model_span[2] == gold_span[2]
                   for gold_span in gold_spans):
                matches.append(model_span)

    # Calculate precision, recall, and F1 with confidence intervals
    gold_total = sum(len(gold_spans) for gold_spans in gold_spans_dict.values())
    model_total = sum(len(model_spans) for model_spans in model_spans_dict.values())

    precision, recall, f1, precision_ci, recall_ci, f1_ci = calculate_precision_recall_f1_with_ci(matches, gold_total, model_total)

    print("Relaxed Matching:")
    print("Precision:", precision, ", 95% confidence interval:", precision_ci)
    print("Recall:", recall, ", 95% confidence interval:", recall_ci)
    print("F1 score:", f1, ", 95% confidence interval:", f1_ci)

    
    
###VARIABLES
# Replace "brain_mri_nlp" and "FINAL_MRI_NLP-cbooth" with your actual annotator names
#gold_annotator = "brain_mri_nlp"
#model_annotator = "FINAL_MRI_NLP-cbooth"

###LOAD DATA
#This opens the JSON containing all the dual-annotated task hashes

f = open('scripts/dual.json') 
annotation_dict = json.load(f) 

            
###PRINT DATA FOR DEBUGGING
"""
# Output the data with one line for each session_id
for input_hash, session_data in annotation_dict.items():
    for session_id, session_info in session_data.items():
        if "spans" in session_info:
            for span in session_info["spans"]:
                print(input_hash, session_id, span["start"], span["end"], span["label"])
        else:
            print(input_hash, session_id, "No spans for this session_id")
"""
    
    
app = typer.Typer()

@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)    
   

def main(
    # fmt: off
    ctx: typer.Context,  # this is only used to read additional arguments
    annotations_path: Path = typer.Argument(..., help="Path to the dual annotation data."),
    #gold_annotator: str = typer.Option("", "--gold-annotator", "-p", help="First annotator dataset."),
    #model_annotator: str = typer.Option("", "--model-annotator", "-p", help="Second annotator dataset.")
    gold_annotator: str = typer.Argument(..., help="First annotator dataset."),
    model_annotator: str = typer.Argument(..., help="Second annotator dataset."),
   ):   

    
    output_standard_matching(annotation_dict, gold_annotator, model_annotator)
    output_relaxed_matching(annotation_dict, gold_annotator, model_annotator)
    
if __name__ == "__main__":
    app()
