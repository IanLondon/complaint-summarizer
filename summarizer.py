#!/usr/bin/env python
# Summarize each topic generated from nmf_topics.py
import argparse

from gensim.summarization import keywords, summarize

import config
from mongo_setup import mongoclient
from process_text import PostManager

# set up encoding to allow piping unicode to file
import sys
import codecs
sys.stdout=codecs.getwriter('utf-8')(sys.stdout)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Generates keywords or sentences for queried documents in subreddit')
    arg_parser.add_argument('--subreddit', type=str, help='subreddit name (or "all" to get all posts', required=True)
    # arg_parser.add_argument('--topic_id', type=int, help='topic id to summarize', required=True)
    # arg_parser.add_argument('--topic_thresh', type=float, help='threshold for specified topic probability of documents', required=True)
    arg_parser.add_argument('--summary_ratio', type=float, help='document to summary ratio. Smaller means shorter summary.', default=0.2)
    arg_parser.add_argument('--single_doc_len', type=float, help='all individual documents are truncated to N characters', default=2500)

    args = arg_parser.parse_args()

    postman = PostManager(mongoclient, args.subreddit)

    search_words = config.SEARCH_WORDS

    doc_char_limit = 60000
    print 'looking at topic-modeled posts in subreddit "%s"' % args.subreddit
    # print 'using topic prob threshold %f' % args.topic_thresh
    print 'per-topic character limit is roughly %i' % doc_char_limit
    print 'per-post character limit is %i' % args.single_doc_len

    for topic_id in sorted(postman.get_topics()):
        print '\nTopic #%s:\n=============' % topic_id
        # query_mixin = {'postwise.tokens': {'$in': search_words}} #TODO: make query more general
        # query_mixin = {'postwise.topic_distro':{'$elemMatch':{'topic_id':topic_id, 'prob':{'$gt':args.topic_thresh}}}}
        query_mixin = {'postwise.topic_assignment.topic':topic_id}
        doc_id_text_generator = postman.fetch_doc_text_body(document_level='postwise', find_query_mixin=query_mixin)

        concat_txt = ''
        breakout = 0 #dumb infinite loop preventer
        while len(concat_txt) < doc_char_limit:
            if breakout > 9999:
                raise IOError('this should never happen')
            try:
                doc_id, text_body = doc_id_text_generator.next()
            except StopIteration:
                print 'not enough docs found, breaking'
                break
            concat_txt = ' '.join([concat_txt, text_body[:args.single_doc_len]])
            breakout += 1

        print 'used %i concatenated docs for this topic' % breakout
        print 'actual character length of concatenated docs: %i' % len(concat_txt)

        # make sure you have something
        if len(concat_txt) == 0:
            print 'got nothing for this topic'
            continue

        # TODO: make arga
        generate_keywords = True
        generate_sentences = True

        if generate_keywords:
            print '\ngenerating keywords\n------------------------------\n'
            summary = keywords(concat_txt, ratio=args.summary_ratio, split=True, lemmatize=True)
            print ', '.join(summary)
        if generate_sentences:
            print '\ngenerating sentences\n------------------------------\n'
            summary = summarize(concat_txt, split=True, ratio=args.summary_ratio)
            for sentence in summary:
                print ' * ' + sentence

        # it's sentence or keyword depending on --sentence flag
