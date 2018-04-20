import inout
from db import Dbinterface
from db.models import Contrato, Diario_Backlisted
from pln import Preprocessor

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
from sqlalchemy import cast, Numeric
from time import time


##
# utils

def split(spliter, data, labels):
    for train, test in spliter.split(data, labels):
        yield data[test], labels[test]


##
# getting data

appconfig = inout.read_yaml('./appconfig')
stopwords = inout.read_json('./stopwords') + inout.read_json('./stopwords.domain')

dbi = Dbinterface(appconfig['db']['connectionstring'])

print('retrieving data')
with dbi.opensession() as session:
    blacklist = list(session.query(Diario_Backlisted.palavra))
    
    contratos = session.query(Contrato).join(Contrato.predicao)

stopwords += [entry[0] for entry in blacklist]
contratos = [{'_id': contrato.id, 'corpo': contrato.objeto, 'classe': contrato.predicao.classe} for contrato in contratos]

random_state = appconfig['random_state']
n_splits = 2


##
# grouping contratos by classe

classes_contratos = {}
for contrato in contratos:

    if contrato['classe'] not in classes_contratos:
        classes_contratos[contrato['classe']] = []

    classes_contratos[contrato['classe']] += [contrato]


##
# preparing preprocessing and spliting tools

prep_tools = Preprocessor()

# preprocess my stopwords. Scikit will remove stopwords AFTER the tokenization process (and i preprocess my tokens in the tokenization process)
# source: https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/feature_extraction/text.py#L265
stopwords = [prep_tools.stem(prep_tools.strip_accents(prep_tools.lowercase(token))) for token in stopwords]

spliter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


##
# preparing pipelines and parameters grid

preprocessing = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords)),
    ('svd', TruncatedSVD(random_state=random_state)),
    ('normalizer', Normalizer(copy=False))
])
preprocessing.set_params(vectorizer__sublinear_tf=True, svd__n_components=100, vectorizer__tokenizer=prep_tools.build_tokenizer())
clusterizer = Pipeline([
    ('clusterizer', MiniBatchKMeans(random_state=random_state))
])

param_grid = ParameterGrid({
    'clusterizer__n_clusters': list(range(5, 51))
})


##
# tuning parameters

print('starting tests')
print('-' * 40)

results = []

total_iterations = len(param_grid) * len(classes_contratos)
index = 0

for classe, documents in classes_contratos.items():

    class_results = { 'class': classe, 'results': [] }

    corpus = [document['corpo'] for document in documents]

    for params in param_grid:
        index += 1

        print('test {}/{} - class: {}'.format(index, total_iterations, classe))
        pprint(params)

        start = time()

        # distribute params between preprocessing and clustering
        preprocessing_params = {i:params[i] for i in params if not i.startswith('clusterizer')}
        clusterizer_params = {i:params[i] for i in params if i.startswith('clusterizer')}

        # update pipelinesparameters
        updated_preprocessing = clone(preprocessing).set_params(**preprocessing_params)
        updated_clusterizer = clone(clusterizer).set_params(**clusterizer_params)

        # preprocess corpus and clusterize
        preprocessed_docs = updated_preprocessing.fit_transform(corpus)
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
        class_results['results'] += [result]

        print('\nsilhouette score: {} (+- {})'.format(round(result['score']['silhouette'], 2), round(result['score']['std_silhouette'], 4)))
        print('calinski & harabaz score: {} (+- {})'.format(round(result['score']['calinski_harabaz'], 2), round(result['score']['std_calinski_harabaz'], 4)))
        print('\nelapsed time: {} seconds'.format(round(time() - start)))
        print('-' * 40)

    results += [class_results]

inout.write_json('./temp/results', results)
