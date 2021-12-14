This repository contains a diagnostic dataset of properties, concepts, and relations. 

The dataset characterizes property-concept pairs in terms of their relations. An overview of relations and examples are shown below:



![](../images/relations.png)


Property-concept pairs have been annotated by means of a crowd annotation task. The task, framework, and an early version of the dataset are introduced in the following publications:


@inproceedings{Sommerauer:etal:2020,
    title = "Would you describe a leopard as yellow? Evaluating crowd-annotations with justified and informative disagreement",
    author = "Sommerauer, Pia  and
      Fokkens, Antske  and
      Vossen, Piek",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.422",
    doi = "10.18653/v1/2020.coling-main.422",
    pages = "4798--4809",
}

@inproceedings{Sommerauer:2020,
    title = "Why is penguin more similar to polar bear than to sea gull? Analyzing conceptual knowledge in distributional models",
    author = "Sommerauer, Pia",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-srw.18",
    pages = "134--142",

}

@inproceedings{sommerauer:etal:2019,
  title={Towards Interpretable, Data-derived Distributional Semantic Representations for Reasoning: A Dataset of Properties and Concepts},
  author={Sommerauer, Pia and Fokkens, Antske and Vossen, Piek},
  booktitle={Wordnet Conference},
  pages={85},
  year={2019}
}

A pilot version of the dataset was presented in:


@inproceedings{sommerauer2018,
  title={Firearms and Tigers are Dangerous, Kitchen Knives and Zebras are Not: Testing whether Word Embeddings Can Tell},
  author={Sommerauer, Pia and Fokkens, Antske},
  booktitle={Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP},
  pages={276--286},
  year={2018}
}



Raw, anonymised crowd annotations (pilot blackbox and diagnostic): `data/raw_anonymised/diagnostic_dataset/`

Cleaned crowd annotations (according to procedure introduced in Sommerauer et. al 2020): `clean_anonymised/diagnostic_dataset/annotations_clean_contradictions_batch_0.5`


Aggregated data (pilot blackbox and diagnostic): `data/aggregated`

Candidate data for annotation: `data/candidate`



