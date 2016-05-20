#!/usr/bin/env python
# Load reddit comments from MongoDB, preprocess,
# train Latent Dirichlet Allocation, and evaluate the model.
from gensim.models.ldamulticore import LdaMulticore
import pymongo
import argparse
import config

class PostFetcher(object):
    """Convenience class for fetching scraped posts for a given subreddit."""
    def __init__(self, mongoclient, subreddit, db=None):
        self.mongoclient = mongoclient
        self.subreddit = subreddit
        self.db = db or config.DEFAULT_DB
        self.posts_collection = self.mongoclient[self.db][config.POSTS_COLLECTION]
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

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Scrapes then streams posts from given subreddit to MongoDB')
    arg_parser.add_argument('--subreddit', type=str, help='subreddit name (or "all" to get all posts')
    arg_parser.add_argument('--db', type=str, help='name of MongoDB database to persist posts to')
    args = arg_parser.parse_args()

    if not args.subreddit:
        raise ValueError('subreddit is required.')

    mongoclient = pymongo.MongoClient()

    fetcher = PostFetcher(mongoclient, args.subreddit, args.db)

    corpus = fetcher.fetch_posts(how='comments_as_docs', min_comments=10)

    # XXX: DEBUG. Print the first few documents
    for document in list(corpus)[:10]:
        print document
        print '\n***********'
