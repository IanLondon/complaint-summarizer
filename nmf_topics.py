import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import config
from mongo_setup import mongoclient
from process_text import PostManager

def print_top_words(model, vectorizer, n_top_words=20, show_vals=False):
    for topic_idx, topic_words in enumerate(word_values(model,vectorizer)):
        print "* Topic %d:" % topic_idx
        if show_vals:
            print ',  '.join(['%s: %.4f' % (word, val) for word, val in topic_words[:n_top_words]])
        else:
            print ' '.join([word for word, _ in topic_words[:n_top_words]])

def word_values(model, vectorizer):
    feature_names = vectorizer.get_feature_names()
    all_words = []

    for topic_idx, topic in enumerate(model.components_):
        all_words.append([(feature_names[i], topic[i]) for i in topic.argsort()[::-1]])

    return all_words

def train_topic_model(text_docs, n_topics, vectorizer_settings=dict(stop_words='english', max_df=0.06, min_df=0.02)):
    """
    Train a tfidf => NMF topic model

    text_docs : array of strings, each representing a document

    n_topics : int
        The number of topics to classify the given documents into
    """
    print 'generating tf-idf matrix'
    vectorizer = TfidfVectorizer(**vectorizer_settings)

    X = vectorizer.fit_transform(text_docs)

    print 'running NMF'
    nmf = NMF(n_components=n_topics).fit(X)

    return nmf, vectorizer

def split_topic(postman, topic_id, n_subtopics):
    """
    If you're splitting a topic into subtopics:
        -select only docs with postwise.topic_assignment.topic == parent_topic_id
        -prepend the parent topic_id to get topics like "3.0", "3.1", "3.2", using a topic_id_namer
    """
    print 'Splitting topic "%s" into %i subtopics' % (topic_id, n_subtopics)

    topic_id_mixin = {'postwise.topic_assignment.topic':topic_id}

    doc_id_text_generator = postman.fetch_doc_text_body(document_level='postwise', find_query_mixin=topic_id_mixin)
    text_docs = [text_body for doc_id, text_body in doc_id_text_generator]

    nmf, vectorizer = train_topic_model(text_docs, n_topics=n_subtopics)

    postman.save_doc_topics_Sklearn(nmf, vectorizer, find_query_mixin=topic_id_mixin,
        topic_id_namer=lambda int_id: '.'.join((topic_id, str(int_id))) )

    print 'split completed'
    print_top_words(nmf, vectorizer)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Generates keywords or sentences for queried documents in subreddit')
    arg_parser.add_argument('--subreddit', type=str, help='subreddit name (or "all" to get all posts', required=True)
    arg_parser.add_argument('--n_topics', type=int, help='number of topics for NMF', required=True)
    arg_parser.add_argument('--min_df', type=float, help='min doc freq for words', required=True)
    arg_parser.add_argument('--max_df', type=float, help='max doc freq for words', required=True)

    args = arg_parser.parse_args()

    postman = PostManager(mongoclient, args.subreddit)

    print 'fetching docs containing SEARCH_WORDS'
    search_words = config.SEARCH_WORDS
    query_mixin = {'postwise.tokens': {'$in': search_words}}

    doc_id_text_generator = postman.fetch_doc_text_body(document_level='postwise', find_query_mixin=query_mixin)
    doc_dict = {doc_id:text_body for doc_id, text_body in doc_id_text_generator}

    vectorizer_settings = dict(stop_words='english', max_df=args.max_df, min_df=args.min_df)

    nmf, vectorizer = train_topic_model(doc_dict.values(),
        n_topics=args.n_topics, vectorizer_settings=vectorizer_settings)

    print_top_words(nmf, vectorizer)

    print 'wiping all topics...'
    postman.wipe_all_topics()
    print 'persisting topics...'
    postman.save_doc_topics_Sklearn(nmf, vectorizer, find_query_mixin=query_mixin)
