import inout
from db import Dbinterface
from db.models import Contrato, Predicao_Contrato, Diario_Backlisted
from pln import Preprocessor

import numpy as np
from functools import reduce
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sqlalchemy import cast, Numeric
from sqlalchemy.sql.expression import bindparam


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


##
# grouping contratos by classe

classes_contratos = {}
for contrato in contratos:

    if contrato['classe'] not in classes_contratos:
        classes_contratos[contrato['classe']] = []

    classes_contratos[contrato['classe']] += [contrato]


##
# preparing preprocessing and clustering tools

prep_tools = Preprocessor()

# preprocess my stopwords. Scikit will remove stopwords AFTER the tokenization process (and i preprocess my tokens in the tokenization process)
# source: https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/feature_extraction/text.py#L265
stopwords = [prep_tools.stem(prep_tools.strip_accents(prep_tools.lowercase(token))) for token in stopwords]

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords, tokenizer=prep_tools.build_tokenizer(), sublinear_tf=True)),
    ('svd', TruncatedSVD(100, random_state=appconfig['random_state'])),
    ('normalizer', Normalizer(copy=False)),
    ('clustering', MiniBatchKMeans(random_state=appconfig['random_state']))
])


##
# clustering routine

print('clustering')
for index, classe in enumerate(appconfig['classification']['allowed_classes']):

    corpus = [contrato['corpo'] for contrato in classes_contratos[classe]]

    pipeline.set_params(clustering__n_clusters=appconfig['clustering']['num_clusters'][index])
    predictions = pipeline.fit_predict(corpus)

    for index, prediction in enumerate(predictions):
        classes_contratos[classe][index]['_cluster'] = np.asscalar(prediction)


##
# persisting

# flatten classes_contratos values
clusterized_contratos = reduce(lambda x,y: x+y, classes_contratos.values())

print('persisting results')
with dbi.opensession() as session:

    predicoes = Predicao_Contrato.__table__

    stmt = predicoes.update() \
        .where(predicoes.c.id == bindparam('_id')) \
        .values(cluster=bindparam('_cluster'))
    session.execute(stmt, clusterized_contratos)

    session.commit()
