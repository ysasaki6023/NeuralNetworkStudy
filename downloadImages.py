#-*- coding:utf-8 -*-
from BeautifulSoup import BeautifulSoup as b
import BeautifulSoup

from urllib import quote_plus as q, unquote_plus as unq, urlencode
from urllib2 import build_opener, urlopen, HTTPCookieProcessor
import urllib2
from cookielib import CookieJar
import urlparse
import os,shutil,httplib2

import urllib2

class GetImages:
    def __init__(self,folderName):
        self.folderName = folderName

    def downloadImages(self,search_word,urls):
        outFolder = os.path.join(self.folderName,search_word)
        if os.path.exists(outFolder)==False:
            os.makedirs(outFolder)
        opener = urllib2.build_opener()
        http = httplib2.Http(".cache")
        for i in range(len(set(urls))):
            try:
                fn, ext = os.path.splitext(urls[i])
                ext = ".jpg"
                response, content = http.request(urls[i])
                with open(os.path.join(outFolder,str(i)+ext), 'wb') as f:
                    f.write(content)
            except:
                continue
            if i%10==0:
                print "%s : Downloading images ( %d / %d )" % (search_word, i, len(urls) )
        return

    def searchLinks(self,search_word,maxNum):
        url_base   = "https://www.google.jp"
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers = {'User-Agent':user_agent,}
        googleImageStep = 20

        url_next   = "/search?site=&tbm=isch&q=%s&tbm=isch"%search_word

        allURLs = []

        for i in range(maxNum/googleImageStep):
            search_request = urllib2.Request(url_base+url_next,None,headers)
            search_results = urllib2.urlopen(search_request)
            soup = BeautifulSoup.BeautifulSoup(search_results.read())
            imageURLs = soup.findAll('img')
            for url in imageURLs:
                allURLs.append( url["src"] )
            print "%s : Searched %d images"%(search_word,len(allURLs))

            try:
                url_next = soup.find("a",text=u"次へ").findParent().findParent()["href"]
            except:
                break

        return allURLs

    def getImages(self,searchWords,maxNum=10000):
        if type(searchWords)==type(""): searchWords = [searchWords]
        if len(searchWords)==1:
            search_str = "%s"%searchWords[0]
        else:
            search_str = ""
            for s in searchWords:
                search_str += "%s+"%s
            search_str = search_str[:-1]
        print search_str

        allURLs = self.searchLinks(search_str,maxNum)
        self.downloadImages(search_str,allURLs)

if __name__=="__main__":
    g = GetImages("data")
    f = open("imageList.txt","r")
    for line in f:
        line = line[:-1]
    	print line
        g.getImages(line,5000)



    g.getImages(["犬"],5000)
    g.getImages(["猫"],5000)
    g.getImages(["森"],5000)
    g.getImages(["同窓会"],5000)
    g.getImages(["飲み会"],5000)
    g.getImages(["散歩"],5000)
    g.getImages(["山"],5000)
    g.getImages(["夕食"],5000)
    g.getImages(["釣り"],5000)

