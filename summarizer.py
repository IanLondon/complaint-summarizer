import argparse

from gensim.summarization import keywords

import config
from mongo_setup import mongoclient
from process_text import PostManager

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Generates keywords for queried documents in subreddit')
    arg_parser.add_argument('--subreddit', type=str, help='subreddit name (or "all" to get all posts', required=True)
    arg_parser.add_argument('--topic_id', type=int, help='topic id to summarize', required=True)
    arg_parser.add_argument('--topic_thresh', type=float, help='threshold for specified topic probability of documents', required=True)

    args = arg_parser.parse_args()

    postman = PostManager(mongoclient, args.subreddit)

    complaint_words = config.COMPLAINT_WORDS
    # query_mixin = {'postwise.tokens': {'$in': complaint_words}} #TODO: make query more general
    query_mixin = {'postwise.topic_distro':{'$elemMatch':{'topic_id':args.topic_id, 'prob':{'$gt':args.topic_thresh}}}}
    doc_id_text_generator = postman.fetch_doc_text_body(document_level='postwise', find_query_mixin=query_mixin)

    # print 'getting docs...'
    # doc_txt = {doc_id:text_body for doc_id, text_body in doc_id_text_generator}

    doc_word_limit = 120000
    breakout = 0 #dumb infinite loop preventer
    print 'debug! only doing char limit around %i' % doc_char_limit
    concat_txt = ''
    while len(concat_txt) < doc_char_limit:
        if breakout > 9999:
            raise InfiniteLoopError
        breakout += 1
        doc_id, text_body = doc_id_text_generator.next()
        concat_txt = ' '.join([concat_txt, text_body])

    print 'used %i concatenated docs' % breakout
    print 'actual length: %i' % len(concat_txt)
    summary = keywords(concat_txt, words=50, split=True, lemmatize=True)
    # print '\n'.join(summary)
    print summary
