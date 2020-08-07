# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:34:08 2020

@author: erika
"""
# Import libraries #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing datasets #

datasets = pd.read_excel('HIST_PAINEL_COVIDBR_18mai2020.xlsx') 
x = list(range(1, 84))
x = np.array(x)
x = x.reshape((-1, 1))
y = datasets.loc[0:82, 'casosAcumulado'].values

# Training the values #

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_regression = PolynomialFeatures(degree = 4)
x_poly = poly_regression.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly , y)

# Vizualizando os dados #

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(poly_regression.fit_transform(x)), color = 'blue')
plt.title('Day x Cases of Covid-19')
plt.xlabel('Day')
plt.ylabel('Cases Covid-19')
plt.show()

# Accuracy #

from sklearn.metrics import accuracy_score