import tweepy
from textblob import TextBlob


consumer_key = 'yeOgYQkt2B6MI2Hq0d00f5nRT'
consumer_secret = 'Ovy0ifB9INPNEHxT1g5bpU0z7hwM0bvasupV2vrYbegJDO5SWt'
access_token = '1092839420364054529-wqlXa9X32w3NXjvzzAX6N4qPb3CTRN'
access_token_secret = '7qLC5lHMbX134FAM07VaSGmV4QBn1eKr90rsoqIWDgbJp'


auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Hurricane')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)

