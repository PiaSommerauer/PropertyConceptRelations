from collections import defaultdict, Counter
from nltk import agreement
from statistics import stdev
import utils
import pandas as pd


class Annotation:
    def __init__(self, prop, concept,
                 relation, workerid,
                 description, example_pos,
                 example_neg, answer,
                 seconds, quid):
        self.prop = prop
        self.concept = concept
        self.relation = relation
        self.workerid = workerid
        self.answer = answer
        if seconds != '':
            self.seconds = float(seconds)
        else:
            self.seconds = None
        self.description = description
        self.example_pos = example_pos
        self.example_neg = example_neg
        self.quid = quid


class AnnotationCollection:
    def __init__(self, annotations):
        self.annotations = annotations

    def create_matrix(self):

        matrix = []
        quid_dict = defaultdict(list)
        for a in self.annotations:
            quid = a.quid
            quid_dict[quid].append(a)
        for quid, annotations in quid_dict.items():
            for n, a in enumerate(annotations):
                worker = a.workerid
                answer = a.answer
                row = [worker, quid, answer]
                matrix.append(row)
        return matrix

    def get_alpha(self):
        matrix = self.create_matrix()
        ratingtask = agreement.AnnotationTask(data=matrix)
        alpha = ratingtask.alpha()
        return alpha

    def get_average_seconds(self):
        seconds = [a.seconds for a in self.annotations if a.seconds is not None]
        av_seconds = sum(seconds) / len(seconds)
        return av_seconds

    def get_clean_average_seconds(self):
        seconds = [a.seconds for a in self.annotations if a.seconds is not None]
        av_seconds = sum(seconds) / len(seconds)
        st = stdev(seconds)
        thresh = av_seconds + 2 * st
        seconds_clean = [s for s in seconds if s < thresh]
        av_seconds_clean = sum(seconds_clean) / len(seconds_clean)
        return av_seconds_clean

    def sort_by_unit(self, collection_labels):

        annotations_by_key = defaultdict(list)

        for annotation in self.annotations:
            attributes = []
            for label in collection_labels:
                att = getattr(annotation, label)
                attributes.append(att)
            attributes_tuple = tuple(attributes)
            annotations_by_key[attributes_tuple].append(annotation)
        return annotations_by_key


class Unit(AnnotationCollection):

    def __init__(self, annotations, prop, concept, relation, quid): #, uas_true):
        super().__init__(annotations)
        self.prop = prop
        self.concept = concept
        self.relation = relation
        self.quid = quid
        #self.true = uas_true
        #self.uas_true = self.true
        self.annotations = annotations
        self.descriptions = set([a.description for a in annotations])
        self.workers = [a.workerid for a in self.annotations]
        self.prop_true = self.get_prop_true()
        if self.prop_true > 0.5:
            self.label = True
        else:
            self.label = False
        self.seconds = self.get_average_seconds()

    def get_prop_true(self):
        answers = [a.answer for a in self.annotations]
        answers_true = answers.count('true')
        # answers_false = answers.count('false')
        prop_true = answers_true / len(answers)
        return prop_true


class Pair(AnnotationCollection):
    # TOODOO add vocab dataa
    def __init__(self, units, prop, concept, metric,
                 cut_off=0.5):
        self.units = units
        self.annotations = []
        for u in self.units:
            for a in u.annotations:
                self.annotations.append(a)
        super().__init__(self.annotations)
        self.prop = prop
        self.concept = concept
        self.metric = metric
        self.cut_off = cut_off
        self.relations = [u.relation for u in self.units if u.label is True]
        self.relations_available = [u.relation for u in self.units]
        self.rel_level_mapping = self.get_level_mapping()
        self.levels = self.get_levels()
        # self.relative_label, self.relative_score = self.assign_relative_label(mode='get_max')
        self.ml_label = self.get_ml_label()

        self.hypothesis, self.rel_hyp, self.prop_hyp = self.get_hypothesis()
        # self.relative_hypothesis = self.get_relative_hypothesis()
        self.n_annotations_unit = sum([len(u.annotations) for u in self.units]) / len(self.units)
        self.seconds = self.get_average_seconds()

    def rank_relations(self, metric):
        relations_ranked = Counter()
        for u in self.units:
            relations_ranked[u.relation] = getattr(u, metric)
        return relations_ranked

    def get_level_mapping(self):
        level_rel_dict = dict()
        rel_level_mapping = dict()
        level_rel_dict['all'] = {'typical_of_concept', 'typical_of_property',
                                 'implied_category', 'affording_activity',
                                 'afforded_usual', 'afforded_unusual'}
        level_rel_dict['some'] = {'variability_limited',
                                  'variability_open', 'variability_subcategories'}

        level_rel_dict['few'] = {'rare', 'unusual', 'impossible'}
        level_rel_dict['creative'] = {'creative'}
        for level, rels in level_rel_dict.items():
            for rel in rels:
                rel_level_mapping[rel] = level
        return rel_level_mapping

    def get_levels(self):
        level_scores = defaultdict(list)
        level_top_score = dict()
        mapping = self.rel_level_mapping
        for rel, prop in self.rank_relations('prop_true').items():
            if prop > 0.5:
                level = mapping[rel]
                level_scores[level].append(prop)
        for level, props in level_scores.items():
            level_top_score[level] = max(props)
        return level_top_score

    def get_ml_label(self):

        # in_pos = any([(rel in pos) for rel in self.levels])
        # in_neg = any([(rel in neg) for rel in self.levels])

        levels = tuple(sorted(list(self.levels.keys())))
        if len(levels) == 1:
            label = levels[0]

        elif len(levels) == 0:
            label = None

        else:
            # if levels in unacceptable_labels:
            #   level_pair_dict['none'][p]=pair
            # else:
            score_dict = defaultdict(list)
            rel_scores = self.levels
            scores_rel = [(score, rel) for rel, score in rel_scores.items()]
            top_score, top_rel = max(scores_rel)
            for rel, score in rel_scores.items():
                score_dict[score].append(rel)
            top_rels = score_dict[top_score]
            top_rels = '-'.join(sorted(top_rels))
            label = top_rels

        if label == 'creative':
            label = 'few'
        elif not label is None:
            if '-creative' in label:
                label = label.replace('-creative', '')
            elif 'creative-' in label:
                label = label.replace('creative-', '')

        if label == 'all-few':
            label = None
        elif label == 'all-few-some':
            label = None

        return label

    def get_hypothesis(self):

        represented = {'typical_of_property', 'afforded_usual',
                       'affording_activity', 'variability_limited'}
        hypothesis = False
        rel_hyp = None
        rate_pos = None
        relations_ranked = self.rank_relations(self.metric)
        #relations_ranked_represented = [(prop_true, rel) for
                                        #rel, prop_true in relations_ranked.items()
                                        #if rel in represented]
        relations_ranked_represented = defaultdict(list)
        relations_ranked_not_represented = defaultdict(list)
        for rel, prop_true in relations_ranked.items():
            if rel in represented:
                relations_ranked_represented[prop_true].append(rel)
            elif rel not in represented:
                relations_ranked_not_represented[prop_true].append(rel)
        #relations_ranked_not_represented = [(prop_true, rel) for
         #                               rel, prop_true in relations_ranked.items()
          #                              if rel not in represented]
        if len(relations_ranked_represented) > 0:
            top_prop_repr = max(list(relations_ranked_represented.keys()))
            top_rel_repr = relations_ranked_represented[top_prop_repr]
            #top_prop_repr, top_rel_repr = max(relations_ranked_represented)
        else:
            top_prop_repr, top_rel_rep = 0.0, None
            
        if len(relations_ranked_not_represented) > 0:
            top_prop = max(list(relations_ranked_not_represented.keys()))
            top_rel = relations_ranked_not_represented[top_prop]
        else:
            top_prop, top_rel = 0.0, None

        if top_prop_repr > self.cut_off:
            hypothesis = True
            top_rel = top_rel_repr
            top_prop = top_prop_repr
        else:
            hypothesis = False

        return hypothesis, top_rel, top_prop

    def get_relation_agreement(self):
        d = dict()
        unit_dict = Counter()
        for unit in self.units:
            unit_dict[(unit.relation)] = unit.get_alpha()
        d['min_rel'] = min(unit_dict, key=unit_dict.get)
        d['min'] = unit_dict[d['min_rel']]
        d['max_rel'] = max(unit_dict, key=unit_dict.get)
        d['max'] = unit_dict[d['max_rel']]
        d['mean'] = sum(unit_dict.values()) / len(unit_dict)
        if len(unit_dict.values()) > 1:
            d['stdv'] = stdev(unit_dict.values())
        else:
            d['stdv'] = None
        return d


class Word():
    def __init__(self, lexical_data_dict):
        self.form = lexical_data_dict['word']
        self.cosine = lexical_data_dict['cosine_centroid']
        self.freq = lexical_data_dict['wiki_frequency']


class Concept():
    def __init__(self, concept, lexical_data_dicts):
        self.concept = concept
        self.lexical_data = lexical_data_dicts
        self.properties = set([d['property'] for d in self.lexical_data])
        self.word_forms = []
        words = []
        for d in self.lexical_data:
            word = d['word']
            if word not in words:
                self.word_forms.append(Word(d))
                words.append(word)
        self.fam = lexical_data_dicts[0]['fam']
        self.conc = lexical_data_dicts[0]['conc']
        self.wn_senses = lexical_data_dicts[0]['n_wn_senses']
        self.sources = lexical_data_dicts[0]['sources_str'].split(' ')
        self.categories = lexical_data_dicts[0]['categories_str']
        self.ambiguity_type = lexical_data_dicts[0]['polysemy_type']
        self.wn_conc = lexical_data_dicts[0]['wn_abs_conc']
        self.met = lexical_data_dicts[0]['mipvu']


class RelationSet(AnnotationCollection):
    def __init__(self, relation, units):
        self.units = units
        self.relation = relation
        self.units_pos = [u for u in units if u.label == True]
        self.units_neg = [u for u in units if u.label == False]
        self.annotations = []
        self.annotations_pos = []
        self.annotations_neg = []
        for unit in units:
            self.annotations.extend(unit.annotations)
            if unit.label == True:
                self.annotations_pos.extend(unit.annotations)
            else:
                self.annotations_neg.extend(unit.annotations)
        self.annotations_pos = AnnotationCollection(self.annotations_pos)
        self.annotations_neg = AnnotationCollection(self.annotations_neg)

        super().__init__(self.annotations)

    def get_unit_agreement(self):
        d = dict()
        unit_dict = Counter()
        for unit in self.units:
            unit_dict[(unit.prop, unit.concept)] = unit.get_alpha()
        d['min_unit'] = min(unit_dict, key=unit_dict.get)
        d['min'] = unit_dict[d['min_unit']]
        d['max_unit'] = max(unit_dict, key=unit_dict.get)
        d['max'] = unit_dict[d['max_unit']]
        d['mean'] = sum(unit_dict.values()) / len(unit_dict)
        if len(unit_dict.values()) > 1:
            d['stdv'] = stdev(unit_dict.values())
        else:
            d['stdv'] = None
        return d


class PairSet(AnnotationCollection):
    def __init__(self, pairs):
        self.pairs = pairs
        self.annotations = []
        for p in self.pairs.values():
            for a in p.annotations:
                self.annotations.append(a)
        super().__init__(self.annotations)
        self.units = []
        for pair in self.pairs.values():
            for u in pair.units:
                self.units.append(u)

    def get_overview(self):
        dict_list = []
        # concept	level	relation	#possible_relations	represented	ml_label
        for pair in self.pairs.values():
            d = dict()
            d['concept'] = pair.concept
            d['ml_label'] = pair.ml_label
            d['hypothesis'] = pair.hypothesis
            d['relations'] = pair.relations
            d['#rel_available'] = len(pair.relations_available)
            dict_list.append(d)
        return dict_list

    def get_pair_agreement(self):
        d = dict()
        pair_dict = Counter()
        for p, pair in self.pairs.items():
            pair_dict[p] = pair.get_alpha()
        d['min_pair'] = min(pair_dict, key=pair_dict.get)
        d['min'] = pair_dict[d['min_pair']]
        d['max_pair'] = max(pair_dict, key=pair_dict.get)
        d['max'] = pair_dict[d['max_pair']]
        d['mean'] = sum(pair_dict.values()) / len(pair_dict)
        if len(pair_dict.values()) > 1:
            d['stdv'] = stdev(pair_dict.values())
        else:
            d['stdv'] = None
        return d


class PropCollections():
    def __init__(self):
        self.perceptual = {'red', 'round', 'yellow',
                           'blue', 'green', 'black', 'warm',
                           'hot', 'cold', 'juicy', 'sweet', 'square'}
        self.activities = {'fly', 'swim', 'lay_eggs', 'roll'}
        self.parts = {'made_of_wood', 'wings', 'wheels'}
        self.complex = {'dangerous', 'used_in_cooking'}
        self.mapping = self.get_prop_collection_dict()

    def get_prop_collection_dict(self):

        prop_collection_dict = dict()
        collections = {'perceptual', 'activities', 'parts', 'complex'}
        for coll in collections:
            props = getattr(self, coll)
            for prop in props:
                prop_collection_dict[prop] = coll
        return prop_collection_dict


class PropSet(PairSet):
    def __init__(self, pairs, prop, collection):
        self.pairs = pairs
        self.prop = prop
        super().__init__(self.pairs)
        self.collection = collection
        self.level_pair_dict = self.assign_pairs_to_labels()
        self.level_pair_dict_bin = self.assign_pairs_to_labels(binary=True)
        self.all = PairSet(self.level_pair_dict['all'])
        self.all_some = PairSet(self.level_pair_dict['all-some'])
        self.some_few = PairSet(self.level_pair_dict['few-some'])
        self.some = PairSet(self.level_pair_dict['some'])
        self.few = PairSet(self.level_pair_dict['few'])
        self.creative = PairSet(self.level_pair_dict['creative'])
        self.pos = PairSet(self.level_pair_dict_bin['pos'])
        self.neg = PairSet(self.level_pair_dict_bin['neg'])
        self.nolabel = PairSet(self.level_pair_dict_bin['no-label'])
        self.none = PairSet(self.level_pair_dict[None])

        self.hyp_true = PairSet(dict([(c, p) for c, p in self.pairs.items()
                                      if p.hypothesis > 0.5]))

    def assign_pairs_to_labels(self, binary=False):

        level_pair_dict = defaultdict(dict)
        pos_labels = ['all', 'all-some', 'some', 'few-some']
        neg_labels = ['few']
        if binary == False:
            for p, pair in self.pairs.items():
                label = pair.ml_label
                level_pair_dict[label][p] = pair
        else:
            for p, pair in self.pairs.items():
                label = pair.ml_label
                if label in pos_labels:
                    level_pair_dict['pos'][p] = pair
                elif label in neg_labels:
                    level_pair_dict['neg'][p] = pair
                else:
                    level_pair_dict['no-label'][p] = pair
                
        return level_pair_dict

    def get_av_n_annotations(self):
        av_annotations = sum([pair.n_annotations_unit for pair in self.pairs.values()]) / len(self.pairs)
        return av_annotations

    def show_no_label_info(self):
        only_neg = []
        only_pos = []
        other = []
        for pair in self.pairs.values():
            if pair.ml_label is None:
                if len(pair.relations_available) < 8:
                    if 'rare' in pair.relations_available:
                        only_neg.append((pair.concept, pair.rank_relations()))
                    elif 'implied_category' in pair.relations_available:
                        only_pos.append([pair.concept, pair.rank_relations()])
                    else:
                        other.append([pair.concept, pair.rank_relations()])
                else:
                    other.append([pair.concept, pair.rank_relations()])

        print('only positive options:')
        for c, rel in only_pos:
            print(c)
            print(rel.most_common())
            print()
        print('only negative options:')
        for c, rel in only_neg:
            print(c)
            print(rel.most_common())
        print()
        print('other issue:')
        for c, rel in other:
            print(c)
            print(rel.most_common())


class ConceptSet(PairSet):
    def __init__(self, pairs, concept):
        self.pairs = pairs
        self.concept_str = concept
        self.concept = list(pairs.values())[0].concept
        super().__init__(self.pairs)
        self.pos_examples = PairSet(dict([(c, p) for c, p in self.pairs.items()
                                          if p.ml_label == 'pos']))
        self.neg_examples = PairSet(dict([(c, p) for c, p in self.pairs.items()
                                          if p.ml_label == 'neg']))
        self.no_label = PairSet(dict([(c, p) for c, p in self.pairs.items()
                                      if p.ml_label is None]))


class DataSet(AnnotationCollection):
    def __init__(self, dict_list, lexical_data, metric):
        self.dict_list = dict_list
        self.keys = ['property', 'concept',
                     'relation', 'workerid',
                     'description', 'exampletrue',
                     'examplefalse', 'answer',
                     'duration_in_seconds', 'quid']
        self.lexical_data = lexical_data
        #self.crowd_truth = crowd_truth_data
        self.metric = metric
        self.annotations = self.get_annotations()
        super().__init__(self.annotations)
        self.units = self.get_units()
        self.pairs = self.get_pairs()
        self.pair_set = PairSet(self.pairs)
        self.prop_collection_mapping = PropCollections().mapping
        self.prop_sets = self.get_prop_sets()
        self.concept_sets = self.get_concept_sets()
        self.relation_sets = self.get_relation_sets(rel_type='fine-grained')
        self.relation_sets_coarse = self.get_relation_sets(rel_type = 'coarse-grained')
        self.prop_collection_dict = dict()
        for prop, prop_set in self.prop_sets.items():
            self.prop_collection_dict[prop] = prop_set.collection

    def get_annotations(self):
        annotations = []
        for d in self.dict_list:
            quid = d['quid']
            if 'test' in quid or 'check' in quid:
                continue
            else:
                data = [d[k] for k in self.keys]
                annotations.append(Annotation(*data))
        return annotations

    def get_units(self):
        annotations_by_unit = self.sort_by_unit(('prop', 'concept', 'relation', 'quid'))
        units = []
        for u, annotations in annotations_by_unit.items():
            prop, c, rel, quid = u
            #uas_true = self.crowd_truth[quid]
            units.append(Unit(annotations, prop, c, rel, quid)) #, uas_true))
        return units

    def get_pairs(self):
        pairs = dict()
        units_by_pair = defaultdict(list)
        for unit in self.units:
            c = unit.concept
            p = unit.prop
            units_by_pair[(p, c)].append(unit)
        for pair_tuple, units in units_by_pair.items():
            p, c = pair_tuple
            lexical_data_dicts = self.lexical_data[c]
            concept = Concept(c, lexical_data_dicts)
            pair = Pair(units, p, concept, self.metric)
            pairs[pair_tuple] = pair
        return pairs

    def get_prop_sets(self):
        prop_sets = dict()
        pairs_by_prop = defaultdict(dict)
        for pair_tuple, pair in self.pairs.items():
            p, c = pair_tuple
            pairs_by_prop[p][pair_tuple] = pair
        for p, pairs in pairs_by_prop.items():
            collection = self.prop_collection_mapping[p]
            prop_set = PropSet(pairs, p, collection)
            prop_sets[p] = prop_set
        return prop_sets

    def get_relation_sets(self, rel_type):
        relation_sets = dict()
        units_by_relation = defaultdict(list)
        
        if rel_type == 'fine-grained':
            for unit in self.units:
                relation = unit.relation
                units_by_relation[relation].append(unit)
        elif rel_type == 'coarse-grained':
            for p, pair in self.pairs.items():
                rel = pair.ml_label
                units = pair.units
                units_by_relation[rel].extend(units)
        for r, units in units_by_relation.items():
            rel = RelationSet(r, units)
            relation_sets[r] = rel
        return relation_sets

    def get_concept_sets(self):
        concept_sets = dict()
        pairs_by_concept = defaultdict(dict)
        for pair in self.pairs.values():
            concept = pair.concept
            pairs_by_concept[concept.concept][(pair.prop, pair.concept.concept)] = pair
        for c, pairs in pairs_by_concept.items():
            concept_sets[c] = ConceptSet(pairs, c)
        return concept_sets

    def get_collection_counts(self, collections='collections'):
        relation_collection_counts = []
        collection_counts = Counter()
        for prop, prop_set in self.prop_sets.items():
            if collections == 'collections':
                collection = prop_set.collection
                collection_counts[collection] += len(prop_set.pairs)
            elif collections == 'properties':
                collection_counts[prop] += len(prop_set.pairs)

        for rel, rel_set in self.relation_sets.items():
            rel_dict = dict()
            rel_dict['relation'] = rel
            coll_counts = Counter()
            pairs = set()
            for u in rel_set.units_pos:
                pairs.add((u.prop, u.concept))
            for prop, concept in pairs:
                if collections == 'collections':
                    coll = self.prop_collection_dict[prop]
                    coll_counts[coll] += 1
                elif collections == 'properties':
                    coll_counts[prop] += 1
            # normalize
            for coll, cnt in coll_counts.items():
                cnt_norm = cnt / collection_counts[coll]
                rel_dict[coll] = cnt_norm
            relation_collection_counts.append(rel_dict)
        return relation_collection_counts

    def get_overview(self, component, concept_attribs=[]):
        dict_list = []
        if component == 'prop':
            data = self.prop_sets
            items = 'concepts'
            item_name = 'pairs'
            i = 1  # index of concept in pair tuple
        elif component == 'concept':
            data = self.concept_sets
            items = 'properties'
            item_name = 'pairs'
            i = 0  # index of prop in pair tuple
        elif component == 'pair':
            items = 'units'
            item_name = 'units'
            data = self.pair_set.pairs
        for com, pair_set in data.items():
            d = dict()
            d[component] = com
            d[f'#{items}'] = len(getattr(pair_set, item_name))
            if component == 'prop':
                d['pos'] = len(pair_set.pos.pairs)
                d['neg'] = len(pair_set.neg.pairs)
                d['no-label'] = len(pair_set.nolabel.pairs)
                d['total-valid'] = d['pos'] + d['neg']
                d['prop-pos'] = d['pos']/d['total-valid']
                #d['distance-balanced'] = 0.5 - d['prop-pos']
                #d['prop-neg'] = d['neg']/d['total-valid']
                #d['#all'] = len(pair_set.all.pairs)
                #d['#all-some'] = len(pair_set.all_some.pairs)
                #d['#some'] = len(pair_set.some.pairs)
                #d['#some-few'] = len(pair_set.some_few.pairs)
                #d['#few'] = len(pair_set.few.pairs)
                #d['#creative'] = len(pair_set.creative.pairs)
                #d['#no_label'] = len(pair_set.none.pairs)
                #d['repre.'] = len(pair_set.hyp_true.pairs)
                d['#annot.'] = pair_set.get_av_n_annotations()
            d['alpha'] = pair_set.get_alpha()
            if component != 'pair':
                agreement_dict = pair_set.get_pair_agreement()
                d['alpha (p mean)'] = agreement_dict['mean']
                d['alpha (p stdev)'] = agreement_dict['stdv']
                d['alpha min'] = agreement_dict['min']
                d['alpha max'] = agreement_dict['max']
                d['pair min'] = agreement_dict['min_pair'][i]
                d['pair max'] = agreement_dict['max_pair'][i]
                d['seconds'] = pair_set.get_clean_average_seconds()
            else:
                agreement_dict = pair_set.get_relation_agreement()
                d['alpha (p mean)'] = agreement_dict['mean']
                d['alpha (p stdev)'] = agreement_dict['stdv']
                d['alpha min'] = agreement_dict['min']
                d['alpha max'] = agreement_dict['max']
                d['pair min'] = agreement_dict['min_rel']
                d['pair max'] = agreement_dict['max_rel']
                d['seconds'] = pair_set.get_clean_average_seconds()
                d['label'] = pair_set.relative_label
                d['relations'] = pair_set.relations
            for a in concept_attribs:
                d[a] = getattr(pair_set.concept, a)
            dict_list.append(d)
        return dict_list

    def get_relation_overview(self, rel_type):
        dict_list = []
        if rel_type == 'fine-grained':
            relation_sets = self.relation_sets
        elif rel_type == 'coarse-grained':
            relation_sets = self.relation_sets_coarse
        for rel, rel_set in relation_sets.items():
            d = dict()
            agreement_dict = rel_set.get_unit_agreement()
            d['relation'] = rel
            d['alpha'] = rel_set.get_alpha()
            d['seconds'] = rel_set.get_clean_average_seconds()
            d['pairs'] = len(rel_set.units_pos) #/ len(rel_set.units)
            d['properties'] = len(set([unit.prop for unit in rel_set.units]))
            d['candidate pairs'] = len(rel_set.units)
            #d['units_neg'] = len(rel_set.units_neg) #/ len(rel_set.units)
#             d['alpha (u mean)'] = agreement_dict['mean']
#             d['alpha (u stdev)'] = agreement_dict['stdv']
#             d['alpha min'] = agreement_dict['min']
#             d['alpha max'] = agreement_dict['max']
#             d['unit min'] = agreement_dict['min_unit']
#             d['unit max'] = agreement_dict['max_unit']
            dict_list.append(d)
        return dict_list

    def get_set_overlaps(self):
        pair_overlaps = dict()
        prop_concepts_dict = dict()
        for prop, prop_set in self.prop_sets.items():
            concepts = set([pair.concept.concept for pair in prop_set.pairs.values()])
            prop_concepts_dict[prop] = concepts
        for prop1 in self.prop_sets:
            for prop2 in self.prop_sets:
                pair = tuple(sorted([prop1, prop2]))
                if pair not in pair_overlaps and len(pair) == 2:
                    overlap = prop_concepts_dict[prop1].intersection(prop_concepts_dict[prop2])
                    pair_overlaps[pair] = overlap
        return pair_overlaps
