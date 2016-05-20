#!/usr/bin/env python
# Scrape past 1000 posts from the given subreddit, including all comments
# and store in MongoDB.
import requests
import logging
import datetime
import sys
import argparse

import praw
from praw.handlers import MultiprocessHandler
import pymongo

import secrets
import config
# document conversion adapted from
# https://gist.github.com/ludar/fe29455bcd121bb79cf9

logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s',
    level=logging.DEBUG,
    filename=config.LOGFILE)
logger = logging.getLogger(__name__)

class MongoRedditStreamer(object):
    """Streams reddit posts into MongoDB"""
    def __init__(self, r, mongoclient, db_name, collection_name, subreddit='all'):
        """
        Arguments
        =========
        r :
        the praw Reddit session, like `r = praw.Reddit()`

        mongoclient:
        An instance of pymongo.MongoClient()

        db_name :
        the name of your MongoDB database

        collection_name :
        the name of the collection used to store comments

        subreddit :
        the name of the subreddit to scrape

        mongoclient_args: go to pymongo.MongoClient
        """
        self.r = r
        self.subreddit = subreddit
        self.client = mongoclient
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
    def convert_to_document(self, post):
        # get comments if there are any
        comments = {}
        if post.num_comments > 0:
            comment_list = praw.helpers.flatten_tree(post.comments, depth_first=True)

            comments = {'comments': [
                {
                    'text': comment.body,
                    # 'author': {
                    #     'id': comment.author.id,
                    #     'name': comment.author.name
                    # },
                    'created': datetime.datetime.fromtimestamp(comment.created)
                }
                for comment in comment_list
            ]}

        post_doc = {
            '_id': post.id,
            'title': post.title,
            # 'author': {
            #     'id': post.author.id,
            #     'name': post.author.name
            # },
            'subreddit': post.subreddit.display_name,
            'text': post.selftext,
            'date': datetime.datetime.fromtimestamp(post.created),
            'num_comments': 0 if not comments else len(comments['comments'])
        }
        post_doc.update(comments)
        return post_doc
    def scrape_to_db(self):
        try:
            new_posts = praw.helpers.submission_stream(self.r, self.subreddit)
            for post in new_posts:
                try:
                    post_doc = self.convert_to_document(post)
                    # upsert if _id (post.id) already exists
                    self.collection.update_one(
                        {'_id':post_doc['_id']},
                        {'$set':post_doc}, upsert=True)
                    logger.info('Saved "%s"' % post.title[:20])
                except requests.exceptions.HTTPError:
                    logger.warning('HTTPError for "%s"' % post.url)
                except AttributeError as err:
                    logger.warning(err)
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    # r.set_oauth_app_info(
    #     client_id=secrets.CLIENT_ID,
    #     client_secret=secrets.SECRET,
    #     redirect_uri='http://127.0.0.1:65010/authorize_callback')

    # use defaults for praw-multiprocess.
    # You have to have the `praw-multiprocess` server running!
    # XXX: This doesn't actually make it faster, since it's a 1:1 praw.Reddit object to thread...
    # but you can scrape multiple subreddits at once
    handler = MultiprocessHandler()
    # handler=None

    r = praw.Reddit('ubuntu:ian-scraper:v0.0.1 (by /u/ian-scraper)', handler=handler)

    arg_parser = argparse.ArgumentParser(description='Scrapes then streams posts from given subreddit to MongoDB')
    arg_parser.add_argument('--subreddit', type=str, help='subreddit name (or "all" to get all posts')
    arg_parser.add_argument('--db', type=str, help='name of MongoDB database to persist posts to')

    args = arg_parser.parse_args()

    if not args.subreddit:
        raise ValueError('subreddit is required.')

    logger.info('{0}\nStarting new scrape process for subreddit: {1}\n{0}'.format('* ' * 12, args.subreddit))

    if not args.db:
        args.db = config.DEFAULT_DB
        logger.info('db not specified, using default: "%s"' % args.db)
    else:
        logger.info('Using db: "%s"' % args.db)

    streamer = MongoRedditStreamer(
        r=r,
        mongoclient=pymongo.MongoClient(),
        db_name=args.db,
        collection_name=config.POSTS_COLLECTION,
        subreddit=args.subreddit
    )

    streamer.scrape_to_db()
