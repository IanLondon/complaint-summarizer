#!/usr/bin/env python
import requests
import logging
import datetime
import sys

import praw
from praw.handlers import MultiprocessHandler
import pymongo

import secrets

# document conversion adapted from
# https://gist.github.com/ludar/fe29455bcd121bb79cf9

LOGFILE = 'scrape.log'

logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s',
    level=logging.DEBUG,
    filename=LOGFILE)
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

    # handler = MultiprocessHandler()
    # use defaults for praw-multiprocess.
    # You have to have the `praw-multiprocess` server running!
    # XXX: This doesn't actually make it faster, since it's a 1:1 praw.Reddit object to thread...

    handler=None
    r = praw.Reddit('ubuntu:ian-scraper:v0.0.1 (by /u/ian-scraper)', handler=handler)

    streamer = MongoRedditStreamer(
        r=r,
        mongoclient=pymongo.MongoClient(),
        db_name='reddit_test',
        collection_name='posts',
        subreddit='arduino'
    )

    streamer.scrape_to_db()
