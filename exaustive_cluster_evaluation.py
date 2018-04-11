import inout
from db import Dbinterface
from db.models import Contrato

import numpy as np
from pprint import pprint
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from time import time


##
# utils

def split(spliter, data, labels):
    for train, test in spliter.split(data, labels):
        yield data[test], labels[test]


##
# get data

stopwords = inout.read_json('./stopwords')
appconfig = inout.read_yaml('./appconfig')

dbi = Dbinterface(appconfig['db']['connectionstring'])

print('retrieving data')
with dbi.opensession() as session:
    contratos = session.query(Contrato)
documents = [contrato.objeto for contrato in contratos]

random_state = appconfig['clustering']['tuning_randstate']
n_splits = 4


##
# prepare pipeline and folding startegy

preprocessing = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords, strip_accents='ascii')),
    ('svd', TruncatedSVD(random_state=random_state)),
    ('normalizer', Normalizer(copy=False))
])
clusterizer = Pipeline([
    ('clusterizer', MiniBatchKMeans(random_state=random_state))
])

param_grid = ParameterGrid({
    'vectorizer__norm': (None, 'l1', 'l2'),
    'vectorizer__sublinear_tf': (True, False),
    'svd__n_components': (100, 250, 500),
    'svd__algorithm': ('randomized', 'arpack'),
    'normalizer__norm': ('l1', 'l2'),
    'clusterizer__n_clusters': (10, 25, 50)
})

spliter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


##
# tune parameters

print('starting tests')
print('-' * 40)

results = []
total_iterations = len(param_grid)
for index, params in enumerate(param_grid):

    print('test {}/{}'.format(index+1, total_iterations))
    pprint(params)

    start = time()

    # distribute params between preprocessing and clustering
    preprocessing_params = {i:params[i] for i in params if not i.startswith('clusterizer')}
    clusterizer_params = {i:params[i] for i in params if i.startswith('clusterizer')}

    # update pipelinesparameters
    updated_preprocessing = clone(preprocessing).set_params(**preprocessing_params)
    updated_clusterizer = clone(clusterizer).set_params(**clusterizer_params)

    # preprocess documents and clusterize
    preprocessed_docs = updated_preprocessing.fit_transform(documents)
    prediction = updated_clusterizer.fit_predict(preprocessed_docs)

    # evaluate results
    score_history = {'silhouette': [], 'calinski_harabaz': []}
    for data, labels in split(spliter, preprocessed_docs, prediction):
        silhouette = silhouette_score(data, labels, metric='euclidean')
        score_history['silhouette'] += [silhouette]

        calinski_harabaz = calinski_harabaz_score(data, labels)
        score_history['calinski_harabaz'] += [calinski_harabaz]

    # store results
    result = {
        'params': params,
        'score_history': score_history,
        'score': {
            'silhouette': np.mean(score_history['silhouette']),
            'std_silhouette': np.std(score_history['silhouette']),
            'calinski_harabaz': np.mean(score_history['calinski_harabaz']),
    		'std_calinski_harabaz': np.std(score_history['calinski_harabaz']),
        }
    }
    results += [result]

    print('\nsilhouette score: {} (+- {})'.format(round(result['score']['silhouette'], 2), round(result['score']['std_silhouette'], 4)))
    print('calinski & harabaz score: {} (+- {})'.format(round(result['score']['calinski_harabaz'], 2), round(result['score']['std_calinski_harabaz'], 4)))
    print('\nelapsed time: {} seconds'.format(round(time() - start)))
    print('-' * 40)

inout.write_json('./temp/results', results)
