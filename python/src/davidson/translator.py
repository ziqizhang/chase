import pandas as pd
import os

print(os.getcwd())
#datain = pd.read_csv("../../../data/annotation/keywordfilered_merged.csv",sep=',', encoding="latin-1", usecols='oft')
#datain = open("../../../data/annotation/tagfilered_merged.csv",mode='r',encoding="latin-1")
#print(datain)

#linedata = []
#line = datain.readline()
#count = 0
#print(line)
#for c in line:
#    if c == ',':
#        linedata.append(line[:count])
#        print(linedata)
#    count = count +1
import csv
import unicodecsv as csvunicode
import re
import string
#csvfile = "../../../data/annotation/tagfilered_merged.csv"
#reader = csv.reader(csvfile, delimiter=',', quotechar='"')
#print(reader)

def translator(file,count):
    with open(file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            tweet = row[1]
            tweet = re.sub(',','',tweet)
            tweet = re.sub('\n', ' ', tweet)
            value = row[0].lower()
            print(value+"-"+row[1])
            if value =="class":
                continue
            if row[0] == '':
                #csvout.writerow(str(count) + "0" * 4 + "2" + tweet)
                output.write(str(count) + ",0" * 4 + ",2," + tweet +","+ row[2]+"\n")
            elif value[0] == 'u'  or value[0]=='x':
                output.write(str(count) + ",0" * 4 + ",2," + tweet +","+ row[2]+ "\n")
            elif value[0] == 'r' or value[0] == 'e' or value[0] == 's' or value[0] == 'y' or value[0]=='i':
                output.write(str(count) + ",0" * 4 + ",0," + tweet +","+ row[2]+"\n")
            else:
                print("-------" + value + tweet)
            count = count + 1
        return count
count = 0


with open("/Users/David/spur/chase/output/output.csv", 'w') as output:
    output.write(",count,hate_speech,offensive_language,neither,class,tweet,id\n")
    count = translator("../../../data/annotation/tagfilered_merged.csv", count)
    count = translator("../../../data/annotation/keywordfilered_merged.csv", count)
    count = translator("../../../data/annotation/unfilered_merged.csv", count)
    with open("../../../data/labeled_data.csv", 'r') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',')
        first = 0
        for row in spamreader:
            if first == 0:
                first = 1
                continue
            tweet = row[6]
            tweet = re.sub(',','',tweet)
            tweet = re.sub('\n', ' ', tweet)
            value = row[5].lower()
            output.write(str(count) + ",0" * 4 + ','+value + ','+ tweet + "\n")
            count = count + 1
    print(count)

'''
count = 0
with open("/Users/David/spur/chase/output/output.csv", 'w') as output:
    output.write(",count,hate_speech,offensive_language,neither,class,tweet\n")
    csvout = csv.writer(output,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    with open("../../../data/annotation/tagfilered_merged.csv", 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            tweet = row[1]
            tweet = re.sub(',','',tweet)
            tweet = re.sub('\n', ' ', tweet)
            value = row[0].lower()
            print(value+"-"+row[1])
            if value =="class":
                continue
            if row[0] == '' or row[0] == 'u':
                #csvout.writerow(str(count) + "0" * 4 + "2" + tweet)
                output.write(str(count) + ",0" * 4 + ",2," + tweet+"\n")
            elif value[0] == 'r' or value[0] == 'e' or value[0] == 's' or value[0] == 'y' or value[0]=='i':
                output.write(str(count) + ",0" * 4 + ",0," + tweet + "\n")
            count = count + 1
    with open("../../../data/annotation/keywordfilered_merged.csv", 'r',encoding='latin-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            tweet = row[1]
            tweet = re.sub(',','',tweet)
            tweet = re.sub('\n',' ',tweet)
            value = row[0].lower()
            print(value+"-"+row[1])
            if value =="class":
                continue
            if row[0] == '' or row[0] == 'u':
                #csvout.writerow(str(count) + "0" * 4 + "2" + tweet)
                output.write(str(count) + ",0" * 4 + ",2," + tweet+"\n")
            elif value[0] == 'r' or value[0] == 'e' or value[0] == 's' or value[0] == 'y' or value[0]=='i':
                output.write(str(count) + ",0" * 4 + ",0," + tweet + "\n")
            count = count + 1
    with open("../../../data/annotation/unfilered_merged.csv", 'r',encoding='latin-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            tweet = row[1]
            tweet = re.sub(',','',tweet)
            tweet = re.sub('\n', ' ', tweet)
            value = row[0].lower()
            print(value+"-"+row[1])
            if value =="class":
                continue
            if row[0] == '' or row[0] == 'u':
                #csvout.writerow(str(count) + "0" * 4 + "2" + tweet)
                output.write(str(count) + ",0" * 4 + ",2," + tweet+"\n")
            elif value[0] == 'r' or value[0] == 'e' or value[0] == 's' or value[0] == 'y' or value[0]=='i':
                output.write(str(count) + ",0" * 4 + ",0," + tweet + "\n")
            count = count + 1
print(count)
'''