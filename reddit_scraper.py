#!/usr/bin/env python
# Scrape past 1000 posts from the given subreddit, including all comments
# and store in MongoDB.
import requests
import logging
import sys
import argparse
import time
from datetime import datetime

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
    def __init__(self, r, mongoclient, db_name, collection_name, subreddit='all', get_historic=False):
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

        mongoclient_args:
            go to pymongo.MongoClient

        get_historic :
            If True, get all posts in subreddit between start of subreddit and now.
            If False, get past ~1000 posts and stream in the new ones.
        """
        self.r = r
        self.subreddit = subreddit
        self.client = mongoclient
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        if get_historic:
            # get all posts from beginning of the subreddit to now
            # start from the last post scraped in same subreddit in the database
            # submissions_between scrapes backwards in time, newer to older
            # That means if you get a single old post, we'll skip to there
            # There's a built-in 2-hour-backward offset

            # TODO: This doesn't work! It just scrapes from most recent!
            # XXX: is submissions_between broken? Try it on its own.
            # oldest_posts = self.collection.find({'subreddit':self.subreddit}).sort('date', pymongo.ASCENDING)
            # try:
            #     oldest_post_date = oldest_posts.next()['date']
            #     oldest_timestamp = time.mktime(oldest_post_date.timetuple())
            #     print 'oldest_timestamp', oldest_timestamp
            #     logger.info('Starting historic scrape from date: %s for subreddit "%s".' % (oldest_post_date, self.subreddit))
            # except StopIteration:
            #     logger.info('No existing posts in subreddit "%s", starting historic scrape from beginning of subreddit.' % self.subreddit)
            #     oldest_timestamp = None
            #
            # self.post_generator = praw.helpers.submissions_between(self.r, self.subreddit, highest_timestamp=oldest_timestamp, newest_first=True, verbosity=2)

            self.post_generator = praw.helpers.submissions_between(self.r, self.subreddit, highest_timestamp=None, newest_first=True)
        else:
            # get past ~1000 posts and stream in new ones
            self.post_generator = praw.helpers.submission_stream(self.r, self.subreddit)

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
                    'created': datetime.fromtimestamp(comment.created)
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
            'subreddit': post.subreddit.display_name.lower(), #lowercase the subreddit names
            'text': post.selftext,
            'date': datetime.fromtimestamp(post.created),
            'num_comments': 0 if not comments else len(comments['comments'])
        }
        post_doc.update(comments)
        return post_doc

    def scrape_to_db(self):
        try:
            new_posts = self.post_generator
            for post in new_posts:
                try:
                    post_doc = self.convert_to_document(post)
                    # upsert: update if _id (post.id) already exists. otherwise, insert.
                    updated = self.collection.update_one(
                        {'_id':post_doc['_id']},
                        {'$set':post_doc},
                        upsert=True)

                    # logger.debug('id:%r ack: %s matched:%s modified:%s' % (post_doc['_id'], updated.acknowledged, updated.matched_count, updated.modified_count))

                    try:
                        post_info_str = 'title:"%s", _id:"%s", date:"%s", subreddit:"%s"' % (post.title[:30], post_doc['_id'], post_doc['date'], post_doc['subreddit'])
                    except KeyError as err:
                        post_info_str = post_doc['_id']
                        logger.warn('Error: %s -- Post _id:"%s" missing keys that I wanted to log.'% (err, post_info_str))

                    if not updated.acknowledged:
                        logger.warn('Failed to add post %s\nraw_result:"%s"' % (post_info_str, str(updated.raw_result)))
                    if updated.matched_count == 1:
                        logger.info('Updated existing post %s' % post_info_str)
                    elif updated.matched_count == 0:
                        logger.info('Added new post %s' % post_info_str)
                    else:
                        logger.critical('wtf? multiple indexes??')

                except requests.exceptions.HTTPError:
                    logger.warning('HTTPError for "%s"' % post.url)

                except AttributeError as err:
                    logger.warning(err)
            logger.info('***FINISHED SCRAPING: No more posts found***')
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
    arg_parser.add_argument('--db', type=str,
        help='name of MongoDB database to persist posts to', default=config.DEFAULT_DB)
    arg_parser.add_argument('--historic', action='store_true',
        help='if included, get all historic posts. Otherwise just stream.')

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
        subreddit=args.subreddit,
        get_historic=args.historic
    )

    streamer.scrape_to_db()
