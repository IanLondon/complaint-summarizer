import argparse

from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora

import config
from mongo_setup import mongoclient
from process_text import PostManager

class LdaProcessor(object):
    def __init__(self, processed_text):
        self.processed_text = processed_text
        self.id2word = corpora.Dictionary(processed_text)
        self.regen_corpus(processed_text)

        print 'Initially got %i words' % len(self.id2word)

    def regen_corpus(self, documents):
        """set or reset the corpus with the given documents"""
        self.corpus = [self.id2word.doc2bow(doc) for doc in documents]
        return self

    def tfidf_filter(self, low_thresh):
        tfidf = TfidfModel(self.corpus, id2word=self.id2word)
        low_value_words = []
        for bow in self.corpus:
            low_value_words += [id for id, value in tfidf[bow] if value < low_thresh]

        self._tfidf = tfidf #save to TfidfModel to the object

        self.id2word.filter_tokens(bad_ids=low_value_words)
        print '%i words left after tf-idf filtering' % len(self.id2word)
        # regenerate corpus with new id2word
        self.regen_corpus(self.processed_text)

        return self

    def train_lda(self, num_topics, **kwargs):
        self.lda = LdaMulticore(self.corpus, id2word=self.id2word, num_topics=num_topics, **kwargs)
        return self

    def word_topics(self, num_words=10):
        return [topic[1] for topic in self.lda.print_topics(num_topics=self.lda.num_topics, num_words=num_words)]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Trains LDA for documents in subreddit')
    arg_parser.add_argument('--subreddit', type=str, help='subreddit name (or "all" to get all posts', required=True)
    arg_parser.add_argument('--num_topics', type=int, help='number of topics for LDA', required=True)
    arg_parser.add_argument('--tfidf_thresh', type=float, help='threshold for filtering out words with lower tf-idf values')
    arg_parser.add_argument('--eta', type=float, help='eta hyperparameter for LDA. Low eta means topics contain more dissimilar words.')
    arg_parser.add_argument('--alpha', type=float, help='alpha hyperparameter for LDA. Low alpha means documents contain more dissimilar topics.')
    args = arg_parser.parse_args()

    postman = PostManager(mongoclient, args.subreddit)

    # corpus is a generator, of lists of word-tokens, for each document
    processed = list(postman.fetch_clean_posts())

    lda_processor = LdaProcessor(processed)

    if args.tfidf_thresh:
        print 'thresholding with tfidf threshold of %f' % args.tfidf_thresh
        lda_processor.tfidf_filter(args.tfidf_thresh)

    lda_kwargs = {lda_arg:vars(args)[lda_arg] for lda_arg in ['eta','alpha']}

    lda_processor.train_lda(args.num_topics, **lda_kwargs)

    print '\n------\n'.join(lda_processor.word_topics())
