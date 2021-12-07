# TODO start code for exploration

from collections import defaultdict
import glob
import csv
import numpy as np
from nltk.corpus import wordnet as wn
import inflect
import os
import json


def sort_by_key(data_dict_list, keys):
    """

    :param data_dict_list: table data (list of dicts)
    :param keys: keys to sort data by
    :return: dict sorting data by keys
    """
    sorted_dict = defaultdict(list)
    for d in data_dict_list:
        if len(keys) == 1:
            key = keys[0] 
            sortkey = d[key]
            if type(sortkey) == str:
                sortkey = sortkey.strip()
        else:
            sortkeys = []
            for key in keys:
                sortkey = d[key].strip()
                sortkeys.append(sortkey)
            sortkey = tuple(sortkeys)
        sorted_dict[sortkey].append(d)
    return sorted_dict


def load_lexical_data():
    data_by_concept = defaultdict(list)
    paths = glob.glob('../data/vocabulary_data/vocab_by_property/*.csv')
    for path in paths:
        with open(path) as infile:
            data_dicts = list(csv.DictReader(infile))
            for d in data_dicts:
                concept = d['concept']
                data_by_concept[concept].append(d)
    return data_by_concept


def load_data(run, group, source):
    if source == 'clean':
        path_dir = f'../data/clean_anonymised/diagnostic_dataset/annotations_clean_contradictions_batch_0.5/'
    elif source == 'raw':
        path_dir = f'../data/raw_anonymised/diagnostic_dataset/'
    path_files = f'{path_dir}run{run}-group_{group}/*.csv'
    print(path_files)
    data = []
    for f in glob.glob(path_files):
        with open(f) as infile:
            data.extend(list(csv.DictReader(infile)))
    return data

def load_crowd_truth(data_iterations, source):
    # e.g. source = clean_contradictions_batch_0.5
    runs = [i[1] for i in data_iterations]
    experiment = 'all'
    
    quid_uas_dict = dict()
    
    # ct is always calculated on entire set
    path_dir = '../data/crowd_truth/results/'
    path_f = f'{path_dir}run{"_".join(runs)}-group_-all--batch-all--{source}-units.csv'
    #run3_4_5_pilot_5_scalar_heat-group_experiment-all--batch-all--data_processed-annotations.csv
    with open(path_f) as infile:
        dict_list = csv.DictReader(infile)
        for d in dict_list:
            pos_resp = d['unit_annotation_score_true']
            quid = d['unit']
            quid_uas_dict[quid] = float(pos_resp)
    return quid_uas_dict

def get_av(pr_list):
    if len(pr_list) > 0:
        av = sum(pr_list)/len(pr_list)
    else:
        av = 0
    return av


def get_synonym_pairs(concepts):
    
    synonyms = dict()
    concept_syns = defaultdict(set)
    
    for c in concepts:
        syns = wn.synsets(c, 'n')
        concept_syns[c].update(syns)

    for c1, syns1 in concept_syns.items():
        for c2, syns2 in concept_syns.items():
            if c1 != c2:
                # check if there is synset overlap
                overlap = syns1.intersection(syns2)
                if overlap:
                    synonym_pair = tuple(sorted([c1, c2]))
                    if synonym_pair not in synonyms:
                        d= dict()
                        d['shared'] = overlap
                        d['#total'] = len(syns1.union(syns2))
                        for c, syn_sets in zip([c1, c2], [syns1, syns2]):
                            d[c] = syn_sets  
                        synonyms[synonym_pair] = d
    return synonyms


def get_centroid(positive_examples, model, pluralize=False):
    engine = inflect.engine()
    matrix = []
    oov = []
    for w in positive_examples:
        # get pl:
        if pluralize:
            plural = engine.plural(w)
            words = [w, plural]
        else:
            words = [w]
        for w in words:
            if w in model.vocab:
                vec = model[w]
                matrix.append(vec)
            else: 
                # pluralize:
                plural = engine.plural(w)
                if plural in model.vocab:
                    vec = model[plural]
                    matrix.append(vec)
                else:
                    oov.append(w)
    matrix = np.array(matrix)
    cent = np.mean(matrix, axis=0)
    return cent, oov
    


def get_distances_to_centroid(centroid, all_concepts, model):
    engine = inflect.engine()
    distance_concept_list = []
    oov = []
    for w in all_concepts:
        if w in model.vocab:
            vec = model[w]
            cosine = np.dot(centroid, vec)/(np.linalg.norm(centroid)*np.linalg.norm(vec))
            distance_concept_list.append((cosine, w, w))
        else:
            plural = engine.plural(w)
            if plural in model.vocab:
                vec = model[plural]
                cosine = np.dot(centroid, vec)/(np.linalg.norm(centroid)*np.linalg.norm(vec))
                distance_concept_list.append((cosine, plural, w))
            else:
                oov.append(w)
            
    return distance_concept_list, oov


def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated/diagnostic_dataset/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties


def get_concept_dict(pair, collection, metric = 'prop_true'):
    concept_dict = dict()
    concept_dict['property_type'] = collection
    info = ['ml_label', 'hypothesis', 'rel_hyp', 'prop_hyp']
    for k in info:
        concept_dict[k] = getattr(pair, k)
    concept_dict['metric'] = metric
    rel_dict = dict()
    rel_ranked = pair.rank_relations(metric = metric)
    for rel in pair.relations_available:
        rel_dict[rel] = rel_ranked[rel]
    concept_dict['relations'] = rel_dict
    return concept_dict


def get_prop_data(prop_set):
    
    prop_dict = dict()
    collection = prop_set.collection
    for (p, c), pair in prop_set.pairs.items():
        prop_dict[c] = get_concept_dict(pair, collection, metric = 'prop_true')
    return prop_dict

def prop_to_file(prop_dict, prop):
    
    path = f'../data/aggregated/{prop}.json'
    with open(path, 'w') as outfile:
        json.dump(prop_dict, outfile, indent = 4)
        
        
def load_prop_data_agg(prop):
    
    path = f'../data/aggregated/diagnostic_dataset/{prop}.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict
