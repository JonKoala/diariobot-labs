import inout
from db import Dbinterface
from db.models import Contrato, Resultado
from pln import Preprocessor

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sqlalchemy import cast, Numeric


# getting data

stopwords = inout.read_json('./stopwords') + inout.read_json('./stopwords.domain')
appconfig = inout.read_yaml('./appconfig')

dbi = Dbinterface(appconfig['db']['connectionstring'])

print('retrieving data')
with dbi.opensession() as session:
    contratos = session.query(Contrato).filter(cast(Contrato.ValorFirmado, Numeric(14,2)) > 100000)

ids = [contrato.id for contrato in contratos]
documents = [contrato.objeto for contrato in contratos]

random_state = 42


# pipelines

prep_tools = Preprocessor()

# preprocess my stopwords. Scikit will remove stopwords AFTER the tokenization process (and i preprocess my tokens in the tokenization process)
# source: https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/feature_extraction/text.py#L265
stopwords = [prep_tools.stem(prep_tools.strip_accents(prep_tools.lowercase(token))) for token in stopwords]

prep = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords, tokenizer=prep_tools.build_tokenizer(), sublinear_tf=True)),
    ('svd', TruncatedSVD(100, random_state=random_state)),
    ('normalizer', Normalizer(copy=False))
])
clusterizer = MiniBatchKMeans(n_clusters=25, random_state=random_state)


# clustering

print('preprocessing')
corpus = prep.fit_transform(documents)
print('clustering')
results = clusterizer.fit_predict(corpus)


# persisting results

print('persisting')
entries = [Resultado(id=ids[index], cluster=np.asscalar(result)) for index, result in enumerate(results)]

with dbi.opensession() as session:

    session.query(Resultado).delete()

    for entry in entries:
        session.add(entry)

    session.commit()
