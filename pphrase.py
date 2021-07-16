from ufal.udpipe import Model, Pipeline
import networkx as nx
import pickle
from compress_fasttext.models import CompressedFastTextKeyedVectors


class Token():

    def __init__(self, token: list):
        self.id = int(token[0])
        self.form = token[1]
        self.lemma = token[2]
        self.upos = token[3]
        self.xpos = token[4]
        self.feats = token[5]
        self.head = int(token[6])
        self.dep = token[7]
        self.misc = token[8]


class Extractor():

    def __init__(self, model):
        self.model = Model.load(model)
        self.pipeline = Pipeline(
            self.model, 'tokenize', Pipeline.DEFAULT,
            Pipeline.DEFAULT, 'conllu')

    def get_preps(self, sent):
        preps = dict()
        fixed = set()
        for tok in sent:
            if not tok.dep =='fixed':
                continue
            if tok.head not in preps:
                preps[tok.head] = [[tok]]
            else:
                preps[tok.head][0].append(tok)
            fixed.add(tok)

        for tok in sent:
            if tok.id in preps:
                preps[tok.id][0].append(tok)
                preps[tok.id].append(tok.head)
            elif tok.upos == 'ADP' and tok not in fixed:
                preps[tok.id] = [[tok], tok.head]

        preps = {k: v for k, v in preps.items()
                 if any([tok.upos == 'ADP' for tok in v[0]])}
        return preps


    def get_succesors_by_id(self, sent, id):
        G = nx.DiGraph()
        G.add_edges_from([(tok.head, tok.id) for tok in sent])
        return nx.dfs_successors(G, source=id)

    def get_sorted_sent(self, tokens):
        return ' '.join([tok.form for tok in sorted(tokens,
                         key=lambda x: x.id, reverse=False)])

    def get_tok_tags(self, token):
        return f'PartOfSpeech={token.upos}|{token.feats}'

    def get_phrase(self, prep, dep_id, sent):
        dep = sent[dep_id-1]
        host = sent[dep.head-1]
        bad_host = False
        members = [dep, host]
        if host == dep or host.upos == 'PUNCT':
            bad_host = True
            members.pop()

        prep_ids = {p.id for p in prep}
        members_ids = {t.id for t in members}

        dep_succesors = self.get_succesors_by_id(sent, dep.id)

        dep_subtree = [sent[i-1] for i in set().union(*dep_succesors.values())
                                       if i not in prep_ids|members_ids]

        phrase = {'phrase': self.get_sorted_sent(prep + dep_subtree + members),
                  'host': host.form if not bad_host else None,
                  'preposition': self.get_sorted_sent(prep),
                  'dependant': dep.form,
                  'full_dependant': self.get_sorted_sent([dep] + dep_subtree),
                  'host_tags': self.get_tok_tags(host) if not bad_host else None,
                  'dependant_tags': self.get_tok_tags(dep),
                  'host_lemma': host.lemma if not bad_host else None,
                  'dependant_lemma': dep.lemma}

        return phrase


    def extract(self, text):

        processed = self.pipeline.process(text)

        print('Extracting phrases...')
        phrases = []
        sent = []
        for line in (processed+'#').splitlines():
            if line.startswith('#') and len(sent):
                preps = self.get_preps(sent)
                for prep, dep_id in preps.values():
                    pphrase = self.get_phrase(prep, dep_id, sent)
                    phrases.append(pphrase)
                sent.clear()
            elif len(line) > 1:
                try:
                    sent.append(
                            Token(
                                line.split('\t')
                            )
                    )
                except ValueError:
                    continue

        return phrases


class Classifier():

    def __init__(self, model, vectorizer):

        self.model = pickle.load(open(model, 'rb'))
        self.vectorizer = CompressedFastTextKeyedVectors.load(vectorizer)

    def preprocess(self, construction: dict):
        return text

    def vectorize(self, text):
        return np.concatenate([self.vectorizer[tok] for tok in text.split()])

    def predict(self):
        return label
