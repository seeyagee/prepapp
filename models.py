from compress_fasttext.models import CompressedFastTextKeyedVectors
import spacy
import pickle
import numpy as np


class Extractor():

    def __init__(self, model='ru_core_news_sm'):
        self.nlp = spacy.load(model)
        self.data_fields = (
            'phrase', 'host', 'prep', 'dependant', 'dependant_morph',
            'dependant_lemma', 'dependant_pos',
            'host_morph', 'host_lemma', 'host_pos')

    def parse(self, text):

        doc = self.nlp(text)

        for tok in doc:
            if tok.pos_ == 'ADP' and tok.dep_ != 'fixed':
                pphrase = []
                pphrase_data = dict.fromkeys(self.data_fields)

                prep = sorted(
                    [tok] + [t for t in tok.children if t.dep_ == 'fixed'],
                    key=lambda x: x.i)
                pphrase.extend(prep)
                pphrase_data['prep'] = ' '.join([t.text for t in prep])

                dep = tok.head
                
                if dep == tok or not dep.morph.get('Case'):
                    continue
                
                full_dep = [dep] + [t for t in dep.children 
                        if t not in prep
                        and t != dep.head]

                pphrase.extend(full_dep)
                pphrase_data['dependant'] = dep.text
                pphrase_data['dependant_morph'] = dep.morph
                pphrase_data['dependant_lemma'] = dep.lemma_
                pphrase_data['dependant_pos'] = dep.pos_

                host = dep.head
                if host != dep:
                    pphrase.append(host)
                    pphrase_data['host'] = host.text
                    pphrase_data['host_morph'] = host.morph
                    pphrase_data['host_lemma'] = host.lemma_
                    pphrase_data['host_pos'] = host.pos_

                pphrase = sorted(pphrase, key=lambda x: x.i)
                pphrase_data['phrase'] = ' '.join([t.text for t in pphrase])
                
                yield pphrase_data


class Classifier():

    def __init__(self, model, vectorizer):

        self.model = pickle.load(open(model, 'rb'))
        self.vectorizer = CompressedFastTextKeyedVectors.load(vectorizer)

    def _vectorize(self, text):
        vec = np.concatenate([self.vectorizer[tok] for tok in text.split()])
        vec = vec.reshape(1, -1)
        return vec

    def predict(self, text):
        vec = self._vectorize(text)
        label = self.model.predict(vec)
        return label
