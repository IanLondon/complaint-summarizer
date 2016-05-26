import argparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF

import config
from mongo_setup import mongoclient
from process_text import PostManager

def print_top_words(model, feature_names, n_top_words=20):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Generates keywords or sentences for queried documents in subreddit')
    arg_parser.add_argument('--subreddit', type=str, help='subreddit name (or "all" to get all posts', required=True)
    arg_parser.add_argument('--num_topics', type=int, help='number of topics for NMF', required=True)
    arg_parser.add_argument('--min_df', type=float, help='min doc freq for words', required=True)
    arg_parser.add_argument('--max_df', type=float, help='max doc freq for words', required=True)
    arg_parser.add_argument('--use_countvectorizer', help='use CountVectorizer instead of TfidfVectorizer.', action="store_true")

    args = arg_parser.parse_args()

    postman = PostManager(mongoclient, args.subreddit)

    complaint_words = config.COMPLAINT_WORDS
    query_mixin = {'postwise.tokens': {'$in': complaint_words}}

    print 'fetching docs...'
    doc_id_text_generator = postman.fetch_doc_text_body(document_level='postwise', find_query_mixin=query_mixin)
    doc_dict = {doc_id:text_body for doc_id, text_body in doc_id_text_generator}

    vectorizer_settings = dict(stop_words='english', max_df=args.max_df, min_df=args.min_df)

    if args.use_countvectorizer:
        print 'generating doc-freq matrix'
        vectorizer = CountVectorizer(**vectorizer_settings)
    else:
        print 'generating tf-idf matrix'
        vectorizer = TfidfVectorizer(**vectorizer_settings)

    X = vectorizer.fit_transform(doc_dict.values())
    feature_names = vectorizer.get_feature_names()


    print 'running NMF'
    nmf = NMF(n_components=args.num_topics).fit(X)

    print_top_words(nmf, feature_names)

    print 'wiping all topics'
    postman.wipe_all_topics()
    print 'persisting topics'
    postman.save_doc_topics_Sklearn(nmf, vectorizer, find_query_mixin=query_mixin)
