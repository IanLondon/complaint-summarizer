# this is crappy, use nmf_topics.py instead.

import argparse

from gensim.models.ldamulticore import LdaMulticore
# from gensim.models.ldamodel import LdaModel
# from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora

import config
from mongo_setup import mongoclient
from process_text import PostManager

class LdaProcessor(object):
    def __init__(self, token_docs, **filter_extremes_args):
        """
        token_docs : a list of lists of word or n-gram or sentence tokens.
            Eg, [['the','crazy','cat'],['that','doggone','dog']]
        """
        self.token_docs = token_docs
        self.id2word = corpora.Dictionary(token_docs)
        if filter_extremes_args:
            print 'filtering words with extreme frequencies'
            self.id2word.filter_extremes(**filter_extremes_args)
        # initialize the bow_corpus
        self.reset_bow_corpus(token_docs)

        print 'Got %i total tokens (words)' % len(self.id2word)

    def reset_bow_corpus(self, documents):
        """set or reset the corpus with the given documents"""
        self.bow_corpus = [self.id2word.doc2bow(doc) for doc in documents]
        return None

    def train_lda(self, num_topics, **kwargs):
        print 'training LDA...'
        self.lda = LdaMulticore(self.bow_corpus, id2word=self.id2word, num_topics=num_topics, **kwargs)
        return self

    def word_topics(self, num_words=10):
        return [topic[1] for topic in self.lda.print_topics(num_topics=self.lda.num_topics, num_words=num_words)]

    # utility functions
    def significant_topic_terms(self, topicid):
        raise NotImplementedError()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Trains LDA for documents in subreddit')
    arg_parser.add_argument('--subreddit', type=str, help='subreddit name (or "all" to get all posts', required=True)
    arg_parser.add_argument('--num_topics', type=int, help='number of topics for LDA', required=True)
    arg_parser.add_argument('--eta', type=float, help='eta hyperparameter for LDA. Low eta means topics contain more dissimilar words.')
    arg_parser.add_argument('--alpha', type=float, help='alpha hyperparameter for LDA. Low alpha means documents contain more dissimilar topics.')
    arg_parser.add_argument('--min_percent', type=float, help='Min percentage of docs that token must appear in to be included', default=0.0)
    arg_parser.add_argument('--max_percent', type=float, help='Max percentage of docs that token must appear in to be included', default=1.0)

    args = arg_parser.parse_args()

    search_words = config.SEARCH_WORDS

    search_words_query_mixin = {'postwise.tokens': {'$in': search_words}}

    postman = PostManager(mongoclient, args.subreddit)

    # corpus is a generator, of lists of word-tokens, for each document
    print 'getting documents from mongo'
    token_docs = list(postman.fetch_doc_tokens(document_level='postwise', find_query_mixin=search_words_query_mixin))
    print 'got %i token_docs documents' % len(token_docs)

    # use filtering here!!
    min_token_freq = args.min_percent * len(token_docs)
    max_token_freq = args.max_percent * len(token_docs)
    # XXX: watchout for this gensim pitfall:
    # no_below takes an absolute number of docs, (min_token_freq)
    # no_above takes a percentage (args.max_percent)
    print 'token freqs\n  min: must appear in at least {0} of {2} docs\n  max: cannot appear in over {1} of {2} docs'.format(min_token_freq, max_token_freq, len(token_docs))


    lda_processor = LdaProcessor(token_docs, no_below=min_token_freq, no_above=args.max_percent)

    complaint_whitelist = lda_processor.id2word.doc2bow(search_words)

    lda_kwargs = {lda_arg:vars(args)[lda_arg] for lda_arg in ['eta','alpha']}

    lda_processor.train_lda(args.num_topics, **lda_kwargs)

    # make new complaint bow with re-trained id2word Dictionary
    complaint_bow = lda_processor.id2word.doc2bow(search_words)

    print '\nthe top words in each topic'
    # print '\n------\n'.join(lda_processor.word_topics())

    for topicid in range(args.num_topics):
        print 'Topic {0} :'.format(topicid)
        print lda_processor.lda.print_topic(topicid, topn=10)
        print '----------'

    print '\n\n=====================\n=====================\n'

    postman.wipe_all_topics()

    # save the topics for all the docs that we selected before
    postman.save_doc_topics_LdaProcessor(lda_processor, find_query_mixin=search_words_query_mixin)
