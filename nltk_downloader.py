#!/usr/bin/env python
# run this script once to install all NLTK dependencies
import nltk

requirements = [
    'stopwords',
    'averaged_perceptron_tagger',
    'punkt',
]

for data_name in requirements:
    nltk.download(data_name)
