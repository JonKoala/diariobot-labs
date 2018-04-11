import inout
from db import Dbinterface
from db.models import Contrato, Resultado

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline


# getting data

stopwords = inout.read_json('./stopwords')
appconfig = inout.read_yaml('./appconfig')

dbi = Dbinterface(appconfig['db']['connectionstring'])

print('retrieving data')
with dbi.opensession() as session:
    contratos = session.query(Contrato)

ids = [contrato.id for contrato in contratos]
documents = [contrato.objeto for contrato in contratos]

random_state = appconfig['clustering']['tuning_randstate']


# pipelines

prep = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords, strip_accents='ascii', sublinear_tf=True)),
    ('svd', TruncatedSVD(100, random_state=random_state)),
    ('normalizer', Normalizer(copy=False))
])
clusterizer = MiniBatchKMeans(n_clusters=50, random_state=random_state)


# clustering

print('preprocessing')
corpus = prep.fit_transform(documents)
print('clustering')
results = clusterizer.fit_predict(corpus)
print('scoring')
print('silhouette: {}'.format(silhouette_score(corpus, results, metric='euclidean', sample_size=20000)))
print('calinski-harabaz: {}'.format(calinski_harabaz_score(corpus, results)))


# persisting results

print('persisting')
entries = [Resultado(id=ids[index], cluster=np.asscalar(result)) for index, result in enumerate(results)]

with dbi.opensession() as session:

    session.query(Resultado).delete()

    for entry in entries:
        session.add(entry)

    session.commit()
