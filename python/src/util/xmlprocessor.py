import csv
import os
from xml.dom import minidom

import re

pattern_num=re.compile(r"^[0-9]+$")

def parse_folder(in_folder, out_file):
    writer=csv.writer(open(out_file,'w'))
    header=["ds","id","count","hate_speech","offensive_language","neither","class","tweet"]
    writer.writerow(header)

    count=0
    for f in os.listdir(in_folder):
        annotations=parse_file(in_folder+"/"+f)
        print(f+str(len(annotations)))
        for key, value in annotations.items():
            if key is not None and len(str(key).strip())>3:
                record=["cs140","","","","","",value,key]
                writer.writerow(record)



def find_containing_tweet(tweets:list, start, end):
    for item in tweets:
        s=item[0]
        e=item[1]
        if start>=s and end <=e:
            return item[2]
    return None



def parse_file(in_file):
    annotations={}
    xmldoc = minidom.parse(in_file)
    text=xmldoc.getElementsByTagName('TEXT')[0].firstChild.nodeValue
    tweets_pattern=re.compile(r'\"(.+?)\"')

    tweets=[]
    for m in tweets_pattern.finditer(text):
        #get both text and offsets
        tweets.append([m.start(),m.end(),m.group().replace('\\n', '').strip()])


    itemlist = xmldoc.getElementsByTagName('Group')
    for s in itemlist:
        offsets=s.attributes['spans'].value
        annotation=s.attributes['hate'].value
        if annotation=='yes':
            annotation='1'
        else:
            annotation='0'

        offsets=offsets.split("~")


        if pattern_num.match(offsets[0]) and pattern_num.match(offsets[1]):
            start=int(offsets[0])
            end=int(offsets[1])

            substring=text[start:end]

            #find tweets
            tweet=find_containing_tweet(tweets, start,end)

            if tweet in annotations.keys():
                ann=annotations[tweet]
                if ann=="no" and annotation=="yes":
                    annotations[tweet]="yes"
            else:
                annotations[tweet]=annotation

    return annotations


parse_folder("/home/zqz/Work/Hate-Speech-ML/GoldStandards",
             "/home/zqz/Work/Hate-Speech-ML/cs140.csv")
