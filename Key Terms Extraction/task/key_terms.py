import string

from lxml import etree
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download('stopwords')


class KeyTermsExtraction:
    def __init__(self, xml_file):
        self.data = etree.parse(xml_file).getroot()

    def get_tokens(self, text):
        tokens = word_tokenize(text.lower())
        return tokens

    def get_lemmatized_tokens(self, tokens):
        lemmas = []
        word_lemmatizer = WordNetLemmatizer()

        for token in tokens:
            lemmas.append(word_lemmatizer.lemmatize(token))

        return lemmas

    def remove_tokens(self, tokens, stopwords=None, punctuation=None, pos=None):
        return [token for token in tokens if token not in stopwords and token not in punctuation and nltk.pos_tag([token])[0][1] == pos]

    def tfidf(self, dataset):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(dataset)
        terms = vectorizer.get_feature_names_out()

        tfidf_matrix = csr_matrix.toarray(tfidf_matrix)
        tfidf_text_tokens = []

        for i in range(len(tfidf_matrix)):
            tfidf_tokens = dict()
            for j in range(len(tfidf_matrix[i])):
                if tfidf_matrix[i][j] != 0.:
                    tfidf_tokens[terms[j]] = tfidf_matrix[i][j]
            tfidf_text_tokens.append(tfidf_tokens)

        return tfidf_text_tokens

    def extraction(self):
        corpus = self.data[0]

        all_noun_tokens_lemmas = []
        all_heads = []
        noun_texts = []

        for news in corpus:
            head = news[0].text
            text = news[1].text

            tokens = self.get_tokens(text)
            tokens_lemmas = self.get_lemmatized_tokens(tokens)
            noun_tokens_lemmas = self.remove_tokens(tokens_lemmas, stopwords=stopwords.words('english'), punctuation=string.punctuation, pos='NN')

            all_heads.append(head)
            all_noun_tokens_lemmas.append(noun_tokens_lemmas)
            noun_texts.append(' '.join(noun_tokens_lemmas))

        tfidf_text_tokens = self.tfidf(noun_texts)

        for head, tfidf_tokens in zip(all_heads, tfidf_text_tokens):
            tfidf_tokens = sorted(tfidf_tokens.items(), key=lambda item: (item[1], item[0]), reverse=True)
            frequent_noun_tokens = [token[0] for token in tfidf_tokens[:5]]

            print(f'{head}:')
            print(' '.join(frequent_noun_tokens))
            print('\n')


key_extractor = KeyTermsExtraction('news.xml')
key_extractor.extraction()
