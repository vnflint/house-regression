#my use case: geographic/price clustering to place a school by the most expensive houses.
#estimating can be done with knn, best with classifying, linear regression is best for others

import matplotlib.pyplot as plt
import seaborn as sn; sn.set() 
import pandas as pd
import numpy as np
pd.set_option("display.max_rows", None, "display.max_columns", None)


houses=pd.read_csv("raw_house_data.csv", delimiter=',', quotechar='"')

#adjust columns so missing values are numerical 
houses=houses.replace(to_replace={"lot_acres":"NaN", "HOA":"None","bathrooms":"None","sqrt_ft":"None","garage":"None", "fireplaces":" "}, value="0")  #replaces missing numeric values in the given columns with a 0
houses["HOA"]=houses["HOA"].str.replace(",","")     #removes commas from larger values in HOA
data_types={"lot_acres":float, "HOA":float, "bathrooms":float, "sqrt_ft":float, "garage":float, "fireplaces":float}     #sets up a dictionary of data types to change numeric columns into numeric forms
houses=houses.astype(data_types)        #applies the data type dictionary

#drop rows with nonsensical missing values
houses=houses.drop(houses[houses["bathrooms"]==0.0].index)
houses=houses.drop(houses[houses["sqrt_ft"]==0.0].index)
houses=houses.drop(houses[houses["lot_acres"]==0.0].index)


houses = houses.reset_index()   #resets the index, which the drop function had created gaps in

#attempt to plot the lat/long locations over a map of the city, with colored dots
#representing the house prices
im = plt.imread('tucson map.png')
plt.figure(figsize = (10,5))
implot = plt.imshow(im)
plt.scatter(houses['longitude'],houses['latitude'], c=houses.sold_price, s=1, alpha = .5)
plt.ylim(32,32.65)
plt.xlim(-111.4, -110.4)
plt.ylabel("Latitude")
plt.xlabel("Longitude")
plt.title("Tucson Housing Prices by Location")



#converts prices to price per square foot
price = np.zeros(len(houses))
for i in range(len(houses)):
    price[i] = houses.sold_price[i]/houses.sqrt_ft[i]
    
    
#bins the prices
prices = np.arange(min(price),max(price),(max(price)-min(price))/(len(price)*2))
prices = np.reshape(prices, (len(houses),2))



#combine latitude and longitude into a vector for each house
loc = np.array([houses.latitude,houses.longitude])
loc = loc.T


class LinearRegression():
    def fit(self, X,y):
        self.w = np.linalg.solve(X.T@X, X.T@y)
        
    def predict(self, X):
        
        return np.matmul(X, self.w)
    


class KNNRegressor():
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X, K, epsilon = 1e-3):
        N = len(X)
        y_hat = np.zeros(N)
                
        for i in range(N):
            dist2 = np.sum((self.X-X[i])**2, axis = 1)
            idxt = np.argsort(dist2)[:K]
            gamma_k = np.exp(-dist2[idxt])/np.exp(-dist2[idxt]).sum()
            y_hat[i] = gamma_k.dot(self.y[idxt])
            
        return y_hat



#knn = KNNRegressor()
#knn.fit(prices, loc)
#y_hatknn = knn.predict(prices, 100)

LR = LinearRegression()
LR.fit(prices, loc)
testprice = np.sort(prices,axis=0)
testprice = testprice[::-1]
y_hat = LR.predict(testprice)


#calculating an R squared with location as y
r2 = 1-np.sum((loc-y_hat)**2)/np.sum((loc-np.mean(loc))**2)