import inout
from classification import Classifier, Dataset, DatasetEntry
from db import Dbinterface
from db.models import Diario_Classificacao, Diario_Backlisted, Contrato, Predicao_Contrato
from pln import Preprocessor

import numpy as np
import re
from sqlalchemy import cast, Numeric


##
# utils

def remove_numbers(text):
    return re.sub(r'\S*\d\S*', ' ', text)


##
# getting data

appconfig = inout.read_yaml('./appconfig')
stopwords = inout.read_json('./stopwords')
classifier_params = inout.read_json(appconfig['classification']['params_filepath'])

dbi = Dbinterface(appconfig['db']['connectionstring'])

print('retrieving data')
with dbi.opensession() as session:

    blacklist = list(session.query(Diario_Backlisted.palavra))

    # get crowdsourced data
    training_dataset = session.query(Diario_Classificacao).filter(Diario_Classificacao.classe_id.in_(appconfig['clustering']['allowed_classes']))
    training_dataset = Dataset([DatasetEntry(publicacao.id, remove_numbers(publicacao.corpo), publicacao.classe_id) for publicacao in training_dataset])

    # get data to predict
    contratos = session.query(Contrato).filter(cast(Contrato.ValorFirmado, Numeric(14,2)) > 100000)


to_predict = [(contrato.id, contrato.objeto) for contrato in contratos]
blacklist = stopwords + [entry[0] for entry in blacklist]


##
# preparing preprocessing and classification tools

prep = Preprocessor()

# preprocess my stopwords (blacklist). Scikit will remove stopwords AFTER the tokenization process (and i preprocess my tokens in the tokenization process)
# source: https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/feature_extraction/text.py#L265
blacklist = [prep.stem(prep.strip_accents(prep.lowercase(token))) for token in blacklist]

domain_params = {'vectorizer__tokenizer': prep.build_tokenizer(), 'classifier__random_state': appconfig['random_state']}
classifier = Classifier({**classifier_params, **domain_params}, blacklist)


##
# classifying

print('classifying contratos')

classifier.train(training_dataset.data, training_dataset.target)

ids, corpus = zip(*to_predict)
predictions = classifier.predict(corpus)
results = zip(ids, predictions)


##
# persisting

print('persisting results')
with dbi.opensession() as session:

    # clean old entries
    session.query(Predicao_Contrato).delete()
    session.flush()

    # insert predicoes
    for result in results:
        predicao = Predicao_Contrato(id=result[0], classe=np.asscalar(result[1]))
        session.add(predicao)

    session.commit()
