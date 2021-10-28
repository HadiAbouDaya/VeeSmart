from selenium import webdriver
import codecs
import os
os.system('cls')

PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)

for i in range(1,236):  #235 is the number of pages to scrape
    completeName = os.path.join('C:\\Users\\hadia\\Desktop\\SUMMER GOALS\\VeeSmart internship\\Webscraping\\WS\\html', ("index{}.txt").format(i))
    file_object = codecs.open(completeName, "w", "utf-8")
    driver.get("https://www.zomato.com/beirut/chilis-ashrafieh/reviews?page={}&amp;sort=dd&amp;filter=reviews-dd".format(i))
    file_object.write(driver.page_source)
    print("Page {} is written.".format(i))      #To make sure that the code is running 

driver.quit()