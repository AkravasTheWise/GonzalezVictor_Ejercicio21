#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[2]:


numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1)) 
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[3]:


clf = LogisticRegression(
    C=50. / len(x_train), penalty='l1', solver='saga', tol=0.1,multi_class='multinomial'
)
clf.fit(x_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(x_test, y_test)

coef = clf.coef_.copy()

plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(8, 8), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel(r'$\beta_%i$' % i)
plt.suptitle('Vectores de coeficientes')
plt.savefig('coeficientes.png')


# In[4]:


matrizDeConfusion=confusion_matrix(y_test,clf.predict(x_test))
#plt.imshow(matrizDeConfusion)
fig, ax = plt.subplots()
ax.matshow(matrizDeConfusion, cmap=plt.cm.Blues)

for i in range(10):
    for j in range(10):
        c = matrizDeConfusion[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.axis('off')
plt.title('Matriz de confusión')
plt.savefig('confusion.png')


# In[ ]:





# In[ ]:




