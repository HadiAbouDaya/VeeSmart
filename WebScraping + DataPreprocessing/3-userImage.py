import os
os.system('cls')
import time
from selenium import webdriver
PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)
from bs4 import BeautifulSoup
import pandas as pd
df = pd.read_csv (r'C:\\Users\\hadia\\Desktop\\SUMMER GOALS\\VeeSmart internship\\Webscraping\\WS\\SourceCode\\Reviews.csv')
images=[]

for link in df['URL']:      #open each link, get the source code, get the first image with class name = ui mini avatar image
    driver.get(link)
    #time.sleep(3)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    driver.quit()
    l = soup.find('img', class_='ui mini avatar image')
    print(soup)
    try:                    #if the user doesn't have an image get the default image
        img = l.attrs['data-original']
    except KeyError:
        img = l.attrs['src']
    images.append(img)

#Add the list as a column to the existing data

df = pd.DataFrame(images)
df.to_csv(r'C:/Users/hadia/Desktop/SUMMER GOALS/VeeSmart internship/Webscraping/WS/SourceCode/userimg.csv',index=False, header=False)