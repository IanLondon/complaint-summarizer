#!/usr/bin/env python
# Load reddit comments from MongoDB, preprocess,
# train Latent Dirichlet Allocation, and evaluate the model.
from gensim.models.ldamulticore import LdaMulticore
import pymongo
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
import re
import string
import argparse
import config

class PostManager(object):
    """Convenience class for fetching scraped posts for a given subreddit."""
    def __init__(self, mongoclient, subreddit, db=None):
        self.mongoclient = mongoclient
        self.subreddit = subreddit
        self.db = db or config.DEFAULT_DB
        self.posts_collection = self.mongoclient[self.db][config.POSTS_COLLECTION]
        self.clean_posts_collection = self.mongoclient[self.db][config.CLEANED_POSTS_COLLECTION]

    def __repr__(self):
        return 'Postmanager(mongoclient={self.mongoclient}, subreddit="{self.subreddit}", db="{self.db}")'.format(self=self)

    def fetch_posts(self, how, min_comments=1):
        """Yields each post scraped from the given subreddit"""
        posts = self.posts_collection.find({
            'subreddit':self.subreddit,
            'num_comments':{'$gt':min_comments}
        })

        if how == 'comments_as_docs':
            return self._comments_as_docs(posts)
        elif how == 'posts_as_docs':
            return self._posts_as_docs(posts)
        else:
            raise ValueError('Unsupported "how" arg: "%s"' % how)

    def _comments_as_docs(self, posts):
        """Returns each individual comment in a post (called by fetch_posts)"""
        for post in posts:
            # first yield the post text body, if any
            if post['text']:
                yield post['text']
            # then yield each comment
            for comment in post['comments']:
                yield comment['text']

    def _posts_as_docs(self, posts):
        """Concatenates all a posts's comments together and yields the result (called by fetch_posts)"""
        for post in posts:
            comments = [comment['text'] for comment in post['comments']]
            # preprend post text body if it exists
            post_text = post['text']
            if post_text:
                comments = [post_text + '\n'] + comments
            yield '\n'.join(comments)

    def persist_cleaned_corpus(self, corpus, drop_existing=False):
        """
        Persist a cleaned corpus to MongoDB.
        Optionally deleting the existing clean documents for the subreddit
        """
        if drop_existing:
            query = {'subreddit':self.subreddit}
            print 'dropping %i existing clean documents for subreddit %s' % (
                self.clean_posts_collection.find(query).count(),
                self.subreddit )

            self.clean_posts_collection.remove(query)

        print 'persisting %i documents to %s for subreddit %s' % (
            len(corpus), self.clean_posts_collection, self.subreddit)

        self.clean_posts_collection.insert_many([
            {
                'body':document,
                'subreddit':self.subreddit
            }
            for document in corpus
        ])
        return None


class Preprocessor(object):
    """
    Convenience class for preprocessing text: tokenizes and filters out words and documents.

    corpus : a list of raw document strings

    min_doc_wordcount & max_doc_wordcount : thresholds for document wordcount.
        Documents with wordcounts outside of these bounds are excluded from the corpus.

    min_word_len & max_word_len : thresholds for word length (no. characters).
        Words with lengths outside these bounds will be dropped.

    stopwords : list of words to exclude

    forbidden_pos_tags : list of Part of Speech (POS) tags to exclude.
        For list of POS tags & definitions, see http://www.clips.ua.ac.be/pages/mbsp-tags

    stem_or_lemma_callback : a callback for stemming or lemmatization function.
        To do neither just pass the default: stem_or_lemma=None
        Warning -- lemmatization might take a long time!
        EG:
            nltk.stem.lancaster.LancasterStemmer().stem
            nltk.stem.WordNetLemmatizer().lemmatize
    """
    def __init__(self, corpus, min_doc_wordcount=0, max_doc_wordcount=float('inf'),
        min_word_len=float('-inf'), max_word_len=float('inf'), stopwords=nltk.corpus.stopwords.words('english'),
        forbidden_pos_tags=[], stem_or_lemma_callback=None, filter_pattern=r'[^a-zA-Z\- ]'):

        self.corpus = list(corpus)
        self.min_doc_wordcount = min_doc_wordcount
        self.max_doc_wordcount = max_doc_wordcount
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.stopwords = stopwords
        self.forbidden_pos_tags = forbidden_pos_tags
        self.stem_or_lemma_callback = stem_or_lemma_callback
        # pattern to replace characters in each word
        # default: remove all non-alpha characters except for hyphen and space
        self.filter_pattern = re.compile(filter_pattern)

    def __repr__(self):
        return 'Preprocessor(corpus, min_doc_wordcount={self.min_doc_wordcount}, max_doc_wordcount={self.max_doc_wordcount}, min_word_len={self.min_word_len}, max_word_len={self.max_word_len}, stopwords={self.stopwords}, forbidden_pos_tags={self.forbidden_pos_tags}, stem_or_lemma_callback={self.stem_or_lemma_callback}), filter_pattern=r"{self.filter_pattern}"'.format(self=self)

    def valid_word(self, word, pos_tag):
        """Evaluates all the conditions that determine whether to keep or discard a word"""
        return (
            word not in self.stopwords
            and (self.max_word_len > len(word) > self.min_word_len)
            and pos_tag not in self.forbidden_pos_tags
        )

    def clean_word(self, word):
        """Lowercase word and remove junk characters using self.filter_pattern"""
        return self.filter_pattern.sub(u'', word.lower())

    def preprocess_doc(self, document):
        """
        Tokenize words in raw-text document
        and remove words that fail to meet criteria
        """
        # tokenize, clean, & tag part-of-speech for all words
        tokens = nltk.word_tokenize(document)
        tagged = nltk.pos_tag(tokens)
        # filter out most invalid words with valid_word()
        processed_document = []
        for word, pos_tag in tagged:
            if self.valid_word(word, pos_tag):
                cleaned_word = self.clean_word(word)
                # things like digits and other junk become empty string,
                # so exclude them from final document
                if cleaned_word:
                    processed_document.append(cleaned_word)

        return processed_document

    def doc_has_valid_wc(self, document):
        """
        Returns True if document length is withing the specified bounds.
        Use on a tokenized document -- otherwise len(document) is character length
            of raw document string, instead of word count!
        """
        if isinstance(document, basestring):
            raise ValueError('Expected tokenized document (an iterable of strings), but got document as a raw string. Tokenize it first!')
        return self.max_doc_wordcount > len(document) > self.min_doc_wordcount

    def perform_stem_or_lem(self, document):
        "Perform stemming or lemmatization on all words in a tokenized document"
        if not self.stem_or_lemma_callback:
            raise ValueError('Called perform_stem_or_lem withouth a stem_or_lemma_callback')
        return [self.stem_or_lemma_callback(word) for word in document]

    def process(self):
        """Master function for preprocessing the documents in self.corpus"""
        # tokenize, then filter & otherwise process words in each document
        # using steps in preprocess_doc()
        pre_corpus_len = len(self.corpus)
        self.corpus = [self.preprocess_doc(doc) for doc in self.corpus]
        # get rid of invalid documents (based on word count)
        self.corpus = [doc for doc in self.corpus if self.doc_has_valid_wc(doc)]
        print 'filtered out %i out of %i documents' % (pre_corpus_len - len(self.corpus), pre_corpus_len)
        # stem or lemmatize
        if self.stem_or_lemma_callback:
            self.corpus = [self.perform_stem_or_lem(doc) for doc in self.corpus]
        # for chaining
        return self

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Scrapes then streams posts from given subreddit to MongoDB')
    arg_parser.add_argument('--subreddit', type=str, help='subreddit name (or "all" to get all posts')
    arg_parser.add_argument('--db', type=str, help='name of MongoDB database to persist posts to')
    args = arg_parser.parse_args()

    if not args.subreddit:
        raise ValueError('subreddit is required.')

    mongoclient = pymongo.MongoClient()

    postman = PostManager(mongoclient, args.subreddit, args.db)

    corpus = postman.fetch_posts(how='comments_as_docs', min_comments=10)

    prepro = Preprocessor(corpus).process()
    clean_corpus = prepro.corpus

    postman.persist_cleaned_corpus(clean_corpus, drop_existing=True)
