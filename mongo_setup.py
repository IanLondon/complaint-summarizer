import pymongo
import config
import secrets
# This MongoClient is re-used by other scripts
if secrets.MONGO_URI:
    print 'connected to secret remote mongo'
    mongoclient = pymongo.MongoClient(secrets.MONGO_URI)
else:
    print 'connected to local MongoDB'
    mongoclient = pymongo.MongoClient()

# convenience function for debugging
def subreddit_counts():
    db = mongoclient[config.DEFAULT_DB]
    posts = db[config.POSTS_COLLECTION]
    clean_posts = db[config.CLEANED_POSTS_COLLECTION]
    print 'raw posts by subreddit\n======================'
    for sub in posts.distinct('subreddit'):
        print '%s\t%i' % (sub, posts.find({'subreddit':sub}).count())
    print '\ncleaned posts by subreddit\n======================'
    for sub in clean_posts.distinct('subreddit'):
        print '%s\t%i' % (sub, clean_posts.find({'subreddit':sub}).count())
