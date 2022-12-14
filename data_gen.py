# Design of the dataset
## Determine an online course will be recommended or not 


### 0. subject = {Design, Social Sciences, Management, Photography, Science, Information Technology, Music, Personal Development}
### 1. subscribers = 0<N(7500,2500)
### 2. free = {0,1} 1 if free with p=0.25,Bernoulli Trial
### 3. fee = 0<(Exp(scale = 100)+10)*10
### 4. reviews = randint(0,subscribers)
### 5. avg reviews = rand.uniform(0.0,5.0)
### 6. level = {Beginner(1), Intermediate(2), Expert(3)}
### 7. letures = 0<N(10,5)
### 8. duration = N(30,10)*letures
### 9. published date = random date(1/1/2001 ~ 12/31/2021 )
### 10. subtitles = {0,1}
### Label-> recommend:{0,1}

## Rule(If 1 of the below rules is satisfied, recommend the course):
### 1. Subscriber > 12000
### 2. review >= 0.8*subscriber && avg.reviews >= 4.5
### 3. level = beginner and fee <= 100
### 4. level = intermediate and fee <= 250
### 5. level = expert and fee <= 500

import random
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import csv
from utils import ruleCheck,simpleRule,jointSimpleRule

num_lecture = 10000
header = ["subject","subscribers","free","fee","reviews","avg reviews","level","lectures","duration","published date","subtitles","recommend"]

# Domain
subject = ['Design', 'Social Sciences', 'Management', 'Photography', 'Science', 'Information Technology', 'Music', 'Personal Development']
level = ['Beginner','Intermediate','Expert']

data = []

# Distribution data pre-generated

subscriber = np.random.normal(7500,2500,num_lecture)
subscriber = np.array(np.round_(subscriber.clip(0)),dtype = "int64")
lectures = np.random.normal(10,5,num_lecture)
lectures = np.array(np.round_(lectures.clip(1)),dtype = "int64")
duration = np.random.normal(30,10,num_lecture)
duration = duration.clip(0)
fee = (np.round_(np.random.exponential(100,num_lecture))+10)*10
fee = np.array(fee,dtype = "int64")

for i in range(num_lecture):
    temp = []
    temp.append(random.choice(subject)) # 0. subject
    temp.append(subscriber[i]) # 1. subscriber
    temp.append(random.choices([0,1],weights=(3,1))[0]) # 2. free
    
    # If free, fee=0
    if temp[2] == 1:
        temp.append(0) #3. fee
    else:
        temp.append(fee[i]) #3. fee
    temp.append(random.randint(0,temp[1])) #4. reviews
    if temp[1] == 0:
        temp.append(0) # If no subscriber, no avg. reviews
    else:
        temp.append(random.randint(0,10)*0.5) #5. avg reviews
    temp.append(random.choice(level)) # 6. Level
    temp.append(lectures[i]) # 7. Number of lectures
    temp.append(round(duration[i]*lectures[i])) #8.  Duration
    start = datetime.date(2002,1,1)
    end = datetime.date(2021,12,31)
    temp.append(start + (end - start) * random.random()) #9. Published Date
    temp.append(random.choice([0,1])) #10. Substitles

    rec,_ = ruleCheck(temp)
    #rec,_ = simpleRule(temp)
    #rec = jointSimpleRule(temp)
    temp.append(rec)
 
    data.append(temp)

with open("data.csv",'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)