"""
This is the code for extracting the annotations from the prodigy database.

This assumes that the initial annotation was under the default session, and the second annotation was under a specific named session_id

v1.0 - 31/07/23 - First working version
v1.1 - 23/08/23 - Extracted this code from agreement.py to be able to run separately

"""
import os
import json
from prodigy.components.db import connect

labels = ["HVL","NO_HVL","RVL","NO_RVL","GVL","NO_GVL"]
db = connect()
dataset = "FINAL_MRI_NLP"
session = 'FINAL_MRI_NLP-SESSIONID'

annotations = db.get_dataset_examples(dataset)
inputhashes = db.get_input_hashes(dataset)

dual_annotation_hashes = []

#for input_hash in inputhashes print each session_id that has an annotation
for input_hash in inputhashes:
    for annotation in annotations:
        if annotation['_input_hash'] == input_hash:
            if annotation['_session_id'] == session:
                dual_annotation_hashes.append(input_hash)

annotation_dict = {}

for input_hash in dual_annotation_hashes:
    annotation_dict[input_hash] = {}
    for annotation in annotations:
        if annotation['_input_hash'] == input_hash:
            annotator = annotation['_session_id']
            annotation_dict[input_hash][annotator] = annotation
            

#To write out to a JSON
with open("dual.json", 'w') as f:
    f.write(json.dumps(annotation_dict))
