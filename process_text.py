#!/usr/bin/env python
# Load reddit posts from MongoDB, preprocess, and add fields of cleaned tokens to each post document
import re
import string
import argparse
import random
from warnings import warn

import config
from mongo_setup import mongoclient

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer

class PostManager(object):
    """Namespace for fetching scraped posts for a given subreddit."""
    def __init__(self, mongoclient, subreddit, read_db=None, write_db=None):
        if not read_db and not write_db:
            read_db = config.DEFAULT_DB
        elif write_db and not read_db:
            raise ValueError('Got write_db, expected a read_db too!')

        if read_db and not write_db:
            # Default: write to the db you read from
            write_db = read_db
        print 'reading from db "%s", writing to db "%s"' % (read_db, write_db)

        self.mongoclient = mongoclient
        self.subreddit = subreddit
        self.read_db = read_db
        self.write_db = write_db

        self.corpus_read = self.mongoclient[self.read_db][config.CORPUS_COLLECTION]
        self.posts_read = self.mongoclient[self.read_db][config.POSTS_COLLECTION]

        self.corpus_write = self.mongoclient[self.write_db][config.CORPUS_COLLECTION]
        self.posts_write = self.mongoclient[self.write_db][config.POSTS_COLLECTION]

    def __repr__(self):
        return 'Postmanager(mongoclient={self.mongoclient}, subreddit="{self.subreddit}", read_db="{self.read_db}", write_db="{self.write_db}")'.format(self=self)

    def fetch_doc_tokens(self, document_level, find_query_mixin={}):
        """
        Generator which yields tokens for the docs which have been processed and tokenized

        required_tokens :
            An optional list of strings
        """
        if document_level != 'postwise':
            raise NotImplementedError('document_level:%s' % document_level)

        query = {'subreddit':self.subreddit, document_level:{'$exists':True}}
        query.update(find_query_mixin)

        for doc in self.posts_read.find(query):
            try:
                yield doc[document_level]['tokens']
            except KeyError:
                # XXX: this shouldn't happen...
                print 'woop, doc missing %s.tokens' % document_level

    def fetch_doc_text_body(self, document_level, find_query_mixin={}):
        """Yields (_id, text_body) for all docs with a concatenated text body field."""
        find_query = {'subreddit': self.subreddit, 'postwise.text':{'$exists':True}}
        find_query.update(find_query_mixin)

        if document_level != 'postwise':
            raise NotImplementedError('document_level:%s' % document_level)

        print 'found %i matching the query for text body docs' % self.posts_read.find(find_query).count()

        for doc in self.posts_read.find(find_query):
            yield doc['_id'], doc[document_level]['text']

    # XXX: Deprecated!
    # def save_doc_topics_LdaProcessor(self, lda_processor, find_query_mixin={}):
    #     """
    #     Uses the trained LDA model in the LdaProcessor to classify all documents in the subreddit,
    #     or all documents returned by the find_query_mixin query dict
    #     """
    #     # XXX: assumes postwise document_level
    #
    #     # add/modify user-specified key:vals to the find_query
    #     find_query = {'subreddit': self.subreddit, 'postwise.tokens':{'$exists':True}}
    #     find_query.update(find_query_mixin)
    #
    #     doc_count = 0
    #     warn('Unlike save_doc_topics_Sklearn which stores topics as strings, save_doc_topics_LdaProcessor stores topics as ints.\nAlso, there is no "postwise.topic_assignment" field')
    #     for doc in self.posts_read.find(find_query):
    #         tokens = doc['postwise']['tokens']
    #         doc_bow = lda_processor.id2word.doc2bow(tokens)
    #         topic_distros = lda_processor.lda.get_document_topics(doc_bow, minimum_probability=0)
    #         topic_dicts = [{'topic_id':topic_id, 'prob':prob} for topic_id, prob in topic_distros]
    #         self.posts_write.update({'_id':doc['_id']},{'$set':{'postwise.topic_distro':topic_dicts}}, upsert=True)
    #         doc_count += 1
    #     print 'Saved topic distros for %i documents' % doc_count

    def merge_topics(self, topic_ids):
        """
        Merges 2 or more topics into a single new topic.
            Just re-assigns all the docs in those topics to new topic.
            Eg merging the topics ["1","2","3"] will go into a new topic named "(1+2+3)".

            You don't need to classify anything!

            Note: topic "prob" information is LOST. Overwritten with an arbitrary number, 1
        """
        new_topic_id = '(' + '+'.join(topic_ids) + ')'
        arbitrary_prob = 1

        result = self.posts_write.update(
            {'subreddit':self.subreddit, 'postwise.topic_assignment.topic':{'$in':topic_ids}},
            {'$set':{'postwise.topic_assignment.topic':new_topic_id,
                'postwise.topic_assignment.prob':arbitrary_prob},
            }, multi=True)

        print 'merged %i documents into "%s"' % (result['nModified'], new_topic_id)

    def save_doc_topics(self, topic_modeler, find_query_mixin={}, topic_id_namer=str):
        """
        Uses the trained TopicModeler to assign
        all docs in the find query to their "strongest" single topic.

        topic_id_namer(int_id) : function which takes an int and maps it to a name.
            Default topic_id_namer is str, so you get: "0", "1", ... N. (Strings)
        """
        nmf = topic_modeler.nmf
        vectorizer = topic_modeler.vectorizer
        # only update docs that are the current subreddit,
        # and have tokens (via process_text.py)
        find_query = {'subreddit': self.subreddit, 'postwise.tokens':{'$exists':True}}
        find_query.update(find_query_mixin)

        doc_count = 0
        for doc in self.posts_read.find(find_query):
            text_body = doc['postwise']['text']

            # Make a dict topic_id:topic_prob, where topic_id is determined by topic_id_namer function
            topic_distros = nmf.transform(vectorizer.transform([text_body]))[0]
            topic_dict = {topic_id_namer(topic_id):prob for topic_id, prob in enumerate(topic_distros)}

            # assign each doc to its most probable topic. In case of a tie, chose a random one.
            # buffer_val allows small variation between two topics to count as a tie,
            # eg 0.0323423 is close enough to 0.0323487
            #
            # individal prob values will shrink as number of topics grow,
            # so the buffer should also shrink as number of topics grow.
            buffer_val = 0.001/(len(topic_distros))
            strongest_topics = [topic_id for topic_id, prob in topic_dict.items()
                if (max(topic_distros) - prob < buffer_val)]

            topic_assignment = random.choice(strongest_topics)

            # DEBUG: introspect the docs with multiple topics
            # if len(strongest_topics) != 1:
            #     print 'DEBUG: Got ambigious topic, choosing randomly.'
            #     print buffer_val
            #     print topic_distros
            #     print strongest_topics
            #     print topic_assignment

            # self.posts_write.update({'_id':doc['_id']},{'$set':{'postwise.topic_distro':topic_dict, 'postwise.topic_assignment':{'topic':topic_assignment,'prob':topic_dict[topic_assignment]}}}, upsert=True)
            #
            # ^ actually, don't persist the topic_distro. Just the assignment.
            self.posts_write.update({'_id':doc['_id']},{'$set':{
                'postwise.topic_assignment':{
                    'topic':topic_assignment,'prob':topic_dict[topic_assignment]}}}, upsert=True)

            doc_count += 1
        print 'Saved topic distros for %i documents' % doc_count

    def wipe_all_topics(self):
        """
        Remove the postwise.topic_distro and postwise.topic_distro fields
        from all documents in the subreddit
         """
        # doc_count = self.posts_read.find({'subreddit':self.subreddit, 'postwise.topic_assignment':{'$exists':True}}).count()
        doc_count = self.posts_write.update({'subreddit':self.subreddit, 'postwise.topic_assignment':{'$exists':True}},
                {'$unset':{'postwise.topic_distro':True,'postwise.topic_assignment':True}}, multi=True)

        print 'wiped topics from %i documents' % doc_count['nModified']

    def get_topics(self):
        topic_ids = self.posts_read.distinct('postwise.topic_assignment.topic', {'subreddit':self.subreddit})

        print '%i topics found' % len(topic_ids)
        return topic_ids

    # def fetch_raw_posts(self, how, min_comments=1):
    #     """
    #     Yields each post scraped from the given subreddit.
    #
    #     how :
    #         'comments_as_docs' or 'posts_as_docs'
    #     """
    #     posts = self.posts_read.find({
    #         'subreddit':self.subreddit,
    #         'num_comments':{'$gt':min_comments}
    #     })
    #
    #     print 'Getting %s, min_comments=%i' % (how, min_comments)
    #
    #     if how == 'comments_as_docs':
    #         return self._comments_as_docs(posts)
    #     elif how == 'posts_as_docs':
    #         return self._posts_as_docs(posts)
    #     else:
    #         raise ValueError('Unsupported "how" arg: "%s"' % how)

    # def generate_corpus(self, min_comments=0):
        # TODO: needs pre-tokenized text
        # """
        # Returns a set of all words in subreddit corpus.
        # Excludes documents with num_comments below min_comments threshold.
        # """
        # posts = self.posts_read.find({
        #     'subreddit':self.subreddit,
        #     'num_comments':{'$gt':min_comments}
        # })
        #
        # return {post.the_words_I_guess_blahhhh for post in self.posts_read}

    # def _comments_as_docs(self, posts):
    #     """Returns each individual comment in a post (called by fetch_raw_posts)"""
    #     for post in posts:
    #         # first yield the post text body, if any
    #         if post['text']:
    #             yield post['text']
    #         # then yield each comment
    #         for comment in post['comments']:
    #             yield comment['text']
    #
    # def _posts_as_docs(self, posts):
    #     """Concatenates all a posts's comments together and yields the result (called by fetch_raw_posts)"""
    #     for post in posts:
    #         comments = [comment['text'] for comment in post['comments']]
    #         # preprend post text body if it exists
    #         post_text = post['text']
    #         if post_text:
    #             comments = [post_text + '\n'] + comments
    #         yield '\n'.join(comments)


def each_comment_from_post(post):
    """
    Yields each individual comment in a post.
    post : a single MongoDB document
    """
    # first yield the post text body, if any
    if post['text']:
        yield post['text']
    # then yield each comment
    for comment in post['comments']:
        yield comment['text']

def all_comments_from_post(post, prepend_title=True):
    """
    Concatenates all a posts's comments together and returns the result
    post : a single MongoDB document
    """
    if 'comments' in post:
        comments = [comment['text'] for comment in post['comments']]
        post_text = post['text']
        title_text = post['title']
        if post_text:
            # preprend post text body if it exists
            comments = [post_text + '\n'] + comments
        if prepend_title and title_text:
            # optionally prepend title
            comments = [title_text] + comments
        return '\n'.join(comments).strip()
    else:
        return ''

class Preprocessor(object):
    """
    Convenience class for preprocessing text: tokenizes and filters out words and documents.

    document_level : specifies what a makes a single document in the corpus
        'commentwise' - each individual comment is a document
        'postwise' - concatenate all comments in a post to make a single document

    min_doc_wordcount & max_doc_wordcount : thresholds for document wordcount.
        Documents with wordcounts outside of these bounds are excluded from the corpus.

    min_word_len & max_word_len : thresholds for word length (no. characters).
        Words with lengths outside these bounds will be dropped.

    stopwords : list of words to exclude

    allowed_pos_tags : list of Part of Speech (POS) tags to include.
        If None, all tags are included
        For list of POS tags & definitions, see http://www.clips.ua.ac.be/pages/mbsp-tags

    stem_or_lemma_callback : a callback for stemming or lemmatization function.
        To do neither just pass the default: stem_or_lemma=None
        Warning -- lemmatization might take a long time!
        EG:
            nltk.stem.lancaster.LancasterStemmer().stem
            nltk.stem.WordNetLemmatizer().lemmatize
    """
    def __init__(self, postman, document_level, min_doc_wordcount=0, max_doc_wordcount=float('inf'),
        min_word_len=float('-inf'), max_word_len=float('inf'), stopwords=nltk.corpus.stopwords.words('english'),
        allowed_pos_tags=None, stem_or_lemma_callback=None, filter_pattern=r'[^a-zA-Z\- ]'):

        # Assign text_generator function depending on document_level
        if document_level not in ['commentwise', 'postwise']:
            raise ValueError('document_level not understood')

        self.postman = postman
        self.document_level = document_level

        self.min_doc_wordcount = min_doc_wordcount
        self.max_doc_wordcount = max_doc_wordcount
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.stopwords = stopwords
        self.allowed_pos_tags = allowed_pos_tags
        self.stem_or_lemma_callback = stem_or_lemma_callback
        # pattern to replace characters in each word
        # default: remove all non-alpha characters except for hyphen and space
        self.filter_pattern = re.compile(filter_pattern)

        self.corpus = set() #this set holds all the words in the documents

    def __repr__(self):
        return 'Preprocessor(document_level="{self.document_level}", min_doc_wordcount={self.min_doc_wordcount}, max_doc_wordcount={self.max_doc_wordcount}, min_word_len={self.min_word_len}, max_word_len={self.max_word_len}, stopwords=stopwords, allowed_pos_tags={self.allowed_pos_tags}, stem_or_lemma_callback={self.stem_or_lemma_callback}), filter_pattern=r"{self.filter_pattern}"'.format(self=self)

    def valid_word(self, word, pos_tag=None):
        """Evaluates all the conditions that determine whether to keep or discard a word"""
        return (
            word not in self.stopwords
            and (self.max_word_len > len(word) > self.min_word_len)
            and (pos_tag==None or self.allowed_pos_tags==None or pos_tag not in self.allowed_pos_tags)
        )

    def clean_word(self, word):
        """Lowercase word and remove junk characters using self.filter_pattern"""
        return self.filter_pattern.sub(u'', word.lower())

    def preprocess_post(self, post):
        """
        Tokenize words in raw-text document bodies
        and remove words that fail to meet word level and document level criteria.

        Then update the posts in MongoDB with new {postwise: [token1, token2, ...]} field
        UNIMPLEMENTED: or new {commentwise: [[tok1a,tok2a], [tok1b,tok2b],...]} field
        """
        # tokenize, clean, & tag part-of-speech for all words
        if self.document_level == 'postwise':

            doc_text = all_comments_from_post(post)
            # leave early if there's nothing there
            if doc_text == '':
                return []

            tokens = nltk.word_tokenize(doc_text)
            # TODO: skip this if there's no POS filtering args!
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
            # finally, update the post
            post['postwise'] = {'tokens': processed_document, 'text': doc_text}
            self.postman.posts_write.update_one({'_id':post['_id']}, {'$set':post}, upsert=True)
        else:
            raise NotImplementedError('document_level: "%s"' % self.document_level)

        return processed_document

    # def doc_has_valid_wc(self, document):
    #     """
    #     Returns True if document length is withing the specified bounds.
    #     Use on a tokenized document -- otherwise len(document) is character length
    #         of raw document string, instead of word count!
    #     """
    #     if isinstance(document, basestring):
    #         raise ValueError('Expected tokenized document (an iterable of strings), but got document as a raw string. Tokenize it first!')
    #     return self.max_doc_wordcount > len(document) > self.min_doc_wordcount

    def perform_stem_or_lem(self, document):
        "Perform stemming or lemmatization on all words in a tokenized document"
        if not self.stem_or_lemma_callback:
            raise ValueError('Called perform_stem_or_lem withouth a stem_or_lemma_callback')
        return [self.stem_or_lemma_callback(word) for word in document]

    def persist_corpus(self):
        """Delete existing corpus (set of unique words) and make a new one."""
        subreddit = self.postman.subreddit
        corpus_coll = self.postman.corpus_write
        subreddit_query = {'subreddit':subreddit}

        preexisting_corpora = corpus_coll.find(subreddit_query).count()
        print 'deleting %i existing corpora for subreddit' % preexisting_corpora
        corpus_coll.delete_many(subreddit_query)

        result = corpus_coll.insert_one({'subreddit':subreddit, 'corpus':list(self.corpus)})
        print 'persisted corpus of length %i' % (len(self.corpus))

        # chaining
        return self

    def process(self):
        """
        Master function for preprocessing documents.
        Reads from postman.posts_read and outputs to postman.posts_write
        """
        # tokenize, then filter & otherwise process words in each document
        # using steps in preprocess_doc()

        all_posts_count = self.postman.posts_read.find({'subreddit': self.postman.subreddit}).count()

        for post_idx, post in enumerate(self.postman.posts_read.find({'subreddit': self.postman.subreddit})):
            # preprocess the post and add the new words to the corpus
            new_words = self.preprocess_post(post)
            self.corpus.update(new_words)

            # print on every Nth post so you know it's alive
            if post_idx % 100 == 0:
                print 'done post %i out of %i' % (post_idx, all_posts_count)

        #TODO:
        print 'word count and other corpus-level filters not implemented, skipping...'
        # corpus-level filtering
        # get rid of invalid documents (based on word count)
        # self.corpus = [doc for doc in self.corpus if self.doc_has_valid_wc(doc)]
        # print 'filtered out %i out of %i documents' % (pre_corpus_len - len(self.corpus), pre_corpus_len)
        # stem or lemmatize
        # if self.stem_or_lemma_callback:
            # self.corpus = [self.perform_stem_or_lem(doc) for doc in self.corpus]
        # for chaining
        #######################################################

        return self

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Scrapes then streams posts from given subreddit to MongoDB')
    arg_parser.add_argument('--subreddit', type=str, help='subreddit name (or "all" to get all posts', required=True)
    arg_parser.add_argument('--read_db', type=str, help='name of MongoDB database to read raw posts from')
    arg_parser.add_argument('--write_db', type=str, help='name of MongoDB database to persist posts to')
    arg_parser.add_argument('--min_comments', type=int, help='minimum number of comments for each post', default=0)
    args = arg_parser.parse_args()

    postman = PostManager(mongoclient, args.subreddit, args.read_db, args.write_db)

    # default: individual comments as docs
    # corpus = postman.fetch_raw_posts(how='posts_as_docs', min_comments=args.min_comments)

    stopwords = nltk.corpus.stopwords.words('english') + ['nt','its']

    # allowed_pos_tags = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','RB','RBR','RBS']
    allowed_pos_tags = ['JJ','JJR','JJS','RB','RBR','RBS'] #just adjectives and adverbs

    prepro = Preprocessor(postman, document_level='postwise', min_doc_wordcount=40,
        min_word_len=3, max_word_len=20, stopwords=stopwords,
        allowed_pos_tags=allowed_pos_tags, stem_or_lemma_callback=None, filter_pattern=r'[^a-zA-Z\- ]')

    # process the raw text and persist to corpus to Mongo
    prepro.process().persist_corpus()
