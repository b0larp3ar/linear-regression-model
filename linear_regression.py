import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

df=pd.read_csv("Customers.csv")


X=df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y=df['Yearly Amount Spent']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=67)

lm=LinearRegression()

lm.fit(X_train, y_train)

predictions=lm.predict(X_test)

sns.scatterplot(x=predictions, y=y_test)
plt.show()