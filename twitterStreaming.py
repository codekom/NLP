from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time

#Get the following from your twitter account here https://developer.twitter.com/
ckey = '****'
csecret = '****'
atoken = '****'
asecret = '****'


class listener(StreamListener):

    def on_data(self,data):
        try:
            print (data)

            saveThis = data
            saveFile = open("twitterTweets4.txt",'a')
            saveFile.write(saveThis)
            saveFile.write('\n')
            saveFile.close()
            return True
        except BaseException:
            print('failed data')
            time.sleep(5)

    def on_error(self,status):
        print (status)


auth = OAuthHandler(ckey,csecret)
auth.set_access_token(atoken,asecret)
twitterStream = Stream(auth,listener())
twitterStream.filter(track=["@cricket"],languages=["en"])
    
