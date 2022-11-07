# Design of the dataset
## Determine an online course will have more than 12500(can be any numbers) subscribers or not

### 0. subscribers = 0<N(7500,2500)
### 1. Subject = {Design, Social Sciences, Management, Photography, Science, Information Technology, Music, Personal Development}
### 2. free = {0,1}
### 3. fee = {x|0<=x<=2000,x=10n}
### 4. reviews = randint(0,subscribers)
### 5. letures = 0<N(10,5)
### 6. level = {Beginner, Intermediate, Expert}
### 7. duration = N(30,10)*letures
### 8. published_timestamp = random datetime (1/1/2001 00:00:00 ~ 12/31/2021 23:59:59)
### 9. avg.reviews = rand.uniform(0.0,5.0)
### 10. substitles = {0,1}

