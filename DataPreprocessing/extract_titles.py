import spacy
import pandas 
import os
import sys
import json 
from spacy.tokenizer import Tokenizer
import glob
import multiprocessing as mp
from joblib import Parallel, delayed
NLP_object = spacy.load("en_core_web_md")


def obtain_title_cleaned_v1(text):
    global NLP_object
    doc = NLP_object(obj['title'])
    punct_removed_title = [ _.lemma_.lower() for _ in doc if not _.is_punct]
    return punct_removed_title

labelled_data_folders = {
    1 : './../../Data_042021/ground_truth/positive/',
    0 : './../../Data_042021/ground_truth/negative/',
}

def get_titles(file):  
    object_dict = {}
    with open(file, 'r') as fh:
        for _line_ in fh.readlines():
            _obj_ = json.loads(_line_)
            _id_ = _obj_['id'] 
            object_dict[_id_] = _obj_ 
            
    text_list = Parallel(n_jobs=mp.cpu_count(), prefer="threads")(delayed(obtain_title_cleaned_v1)(obj['title'], ) for obj in object_dict.values())
    return text_list


# ===========================
# Return titles
# ===========================
def get_labelled_titles():
    labelled_title_dict = {}
    for label, folder in labelled_data_folders.items():
        files = glob.glob(os.path.join(folder, '**.json'))
        title_list = []
        for file in files:
            _results = get_titles(file)
            title_list.extend(_results)
        labelled_title_dict[label] = title_list
    return labelled_title_dict