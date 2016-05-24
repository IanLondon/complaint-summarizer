import pymongo
import config
import secrets
# This MongoClient is re-used by other scripts
try:
    # if there's a MONGO_URI in secrets.py, use that URL
    mongoclient = pymongo.MongoClient(secrets.MONGO_URI)
    print 'connected to secret remote mongo'
except AttributeError:
    # otherwise use localhost default
    mongoclient = pymongo.MongoClient()
    print 'connected to local MongoDB'

# convenience function for debugging
def subreddit_counts():
    db = mongoclient[config.DEFAULT_DB]
    posts = db[config.POSTS_COLLECTION]
    print 'raw posts by subreddit\n======================'
    for sub in posts.distinct('subreddit'):
        print '%s\t%i' % (sub, posts.find({'subreddit':sub}).count())
    # print '\ncleaned posts by subreddit\n======================'
    # for sub in clean_posts.distinct('subreddit'):
    #     print '%s\t%i' % (sub, clean_posts.find({'subreddit':sub}).count())
