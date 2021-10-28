from bs4 import BeautifulSoup
import pandas as pd
from csv import writer
import json
import os
os.system('cls')
print("_"*160,"\n")
print("_"*160)
data=[]
restokeys=[]
dict_filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
resto={}
#159
for i in range(1,236):  #235 is the number of files containing the source code
    PATH = ('C:\\Users\\hadia\\Desktop\\SUMMER GOALS\\VeeSmart internship\\Webscraping\\WS\\SourceCode\\chilis-ashrafieh\\index{}.txt').format(i)
    with open(PATH, encoding="utf8") as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
        f.close()
    script=soup.findAll('script',type="application/ld+json")[1]     #getting the second script tag that contains the restaurant and review data
    json_object = json.loads(script.contents[0])        #creating a dict
    if i==1:        #restaurant specific data is always the same so we get it only one time
        for f in [1,2,3,4,5,10,13,15]:  #1=type | 2=name | 3=URL | 4=openingHours | 5=map | 10=phone number | 13=image | 15=Agg Rating
            restokeys.append(list(json_object.keys())[f])
        resto=dict_filter(json_object, restokeys)
    reviews=json_object.get('reviews')  #get the reviews value which is a list of dict
    for rev in reviews: #get the needed data from each dict
        author = BeautifulSoup(rev.get('author'), "lxml").text
        url = BeautifulSoup(rev.get('url'), "lxml").text
        description = BeautifulSoup(rev.get('description'), "lxml").text
        rating = BeautifulSoup(str(rev.get('reviewRating').get('ratingValue')), "lxml").text
        data.append([resto.get('name'),'https://www.zomato.com'+str(resto.get('url')),author, url, description, rating])    #append it to our main list

print(resto)        #restaurant info

df = pd.DataFrame(data, columns=['Resturant','RestoURL','Author','URL','Review','Rating'])
df.to_csv(r'C:/Users/hadia/Desktop/SUMMER GOALS/VeeSmart internship/Webscraping/WS/SourceCode/Reviews.csv',index=False, header=False)