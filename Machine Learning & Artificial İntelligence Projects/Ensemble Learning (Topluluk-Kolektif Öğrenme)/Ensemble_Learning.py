# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:53:03 2023

@author: Hasan Emre
"""
#%%  import library
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

# warning library
import warnings
warnings.filterwarnings("ignore")


#%% Data Sets

# Veri kumelerinin boyutlari ve daginiklik seviyeleri 
n_samples = 1000
n_features = 2
n_classes = 2

random_state = 42

noise_moon = 0.1
noise_class = 0.2
noise_circle = 0.1

# Veri kumesi 1: make_classification ile siniflandirma veri kumesi olusturma
x, y = make_classification(n_samples=n_samples, 
                    n_features=n_features,
                    n_classes= n_classes,
                    n_repeated=0,
                    n_redundant=0,
                    n_informative= n_features-1,
                    random_state = random_state,
                    n_clusters_per_class=1,
                    flip_y= noise_class)
 
# Visualization: siniflandirma veri kumesini gorsellestir
data = pd.DataFrame(x)
data["target"] = y
plt.figure()
sns.scatterplot(x = data.iloc[:,0], y = data.iloc[:,1], hue="target", data = data)

data_clasification = (x,y)

##
# Dataset 2: make_moons ile yari ay seklinde bir veri kumesi olusturma
moon = make_moons(n_samples = n_samples,
                  noise=noise_moon,
                  random_state=random_state) 

data = pd.DataFrame(moon[0])
data["target"] = moon[1]
plt.figure()
sns.scatterplot(x = data.iloc[:,0], y = data.iloc[:,1], hue="target", data = data)


##
# Dataset 3: make_circles ile daire seklinde bir veri kumesi olusturma
circle = make_circles(n_samples = n_samples,
                      factor=0.5,
                  noise=noise_circle,
                  random_state=random_state) 

# Visualization: daire seklindeki veri kumesini gorsellestir
data = pd.DataFrame(circle[0])
data["target"] = circle[1]
plt.figure()
sns.scatterplot(x = data.iloc[:,0], y = data.iloc[:,1], hue="target", data = data)

# Olusturulan veri kumelerini bir listeye ekleyin
datasets = [moon, circle]

#%% Classification, K-En Yakın Komsu (KNN), Destek Vektor Makinesi (SVM), ve Karar Agacı (DT) modelleri
n_estimators = 10

# Siniflandiricilari olustur
svc = SVC()  # Destek Vektor Makinesi (SVM)
knn = KNeighborsClassifier(n_neighbors=15)  # K-En Yakın Komsu (KNN)
dt = DecisionTreeClassifier(random_state=random_state)  # Karar Agacı (DT)

# Diger modeller: Rastgele Orman (Random Forest), AdaBoost ve Oy Birliği (Voting)
rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)  # Rastgele Orman (Random Forest)
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=n_estimators, random_state=random_state)  # AdaBoost
vt1 = VotingClassifier(estimators=[("svc", svc), ("knn", knn), ("dt", dt), ("rf", rf), ("ada", ada)])  # Oy Birliği (Voting)

# Siniflandiricilarin isimlerini ve listesini belirle
names = ["SVC", "KNN", "Decision Tree", "Random Forest", "AdaBoost", "Voting"]
classifiers = [svc, knn, dt, rf, ada, vt1]


# Meshgrid icin adım boyutu ve sayaclar
h = 0.2
i = 1

# Gorsellestirme için alt yapi
figure = plt.figure(figsize=(18,6))

# Her veri kumesi için dongu
for ds_cnt, ds in enumerate(datasets):
    
    x, y = ds
    x = RobustScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.4, random_state=random_state)
 
    x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
    y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max,h ),
                         np.arange(y_min, y_max, h))
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000","#0000FF"])
    
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    
    if ds_cnt == 0:
        ax.set_title("Input data")
        
    
    ax.scatter(x_train[:,0], x_train[:,1], c = y_train, cmap = cm_bright, edgecolors = "k")
    ax.scatter(x_test[:,0], x_test[:,1], c = y_test, cmap = cm_bright, alpha = 0.6, marker = "^", edgecolors = "k")

    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    print("Dataset # {}".format(ds_cnt))
    
    
    # Her Siniflandirici modeli icin dongu
    for name, clf in zip(names, classifiers):
        
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        
        print("{}: test set score: {} ". format(name, score))
        
        score_train = clf.score(x_train, y_train)
        
        print("{}: train set score: {} ". format(name, score_train))
        print()


        
        if hasattr(clf, "decision_function"):
            z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else: 
            z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        
        
        # Put the result into color plot (Renkli kontur grafigi olustur)
        z = z.reshape(xx.shape)
        ax.contourf(xx, yy, z, cmap = cm, alpha= .8)
        
        # Plot the training points (Egitim verilerini gorsellestir)
        ax.scatter(x_train[:,0], x_train[:,1], c = y_train, cmap = cm_bright, edgecolors = "k")
        
        # Plot the testing points  (Test verilerini gorsellestir)
        ax.scatter(x_test[:,0], x_test[:,1], c = y_test, cmap = cm_bright, alpha = 0.6, marker = "^", edgecolors = "k")
        
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        score = score*100
        ax.text(xx.max() - .3, yy.min() + .3, ("%.1f" %score),
                size = 15, horizontalalignment = "right")
            
        i += 1
    print("---------------------------------------------------")
    
            
plt.tight_layout()
plt.show()
            


def make_classify(dc, clf, name):
    # Veri kumesini al
    x, y = ds
    
    # Verileri RobustScaler ile olceklendir
    x = RobustScaler().fit_transform(x)
    
    # Egitim ve test verilerini bolelim
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state)
    
    # Her siniflandirici modeli icin dongu
    for name, clf in zip(names, classifiers):
        
        # Siniflandiriciyi egit
        clf.fit(x_train, y_train)
        
        # Test veri kumesi uzerinde basarı puanini hesapla ve ekrana yazdir
        score = clf.score(x_test, y_test)
        print("{}: test set score: {}".format(name, score))
        
        # Egitim veri kumesi uzerinde basarı puanini hesapla ve ekrana yazdir
        score_train = clf.score(x_train, y_train)
        print("{}: train set score: {}".format(name, score_train))
        print()

# "make_classify" fonksiyonunu cagirarak "make_classification" veri kumesini siniflandiricilar ile değerlendir
print("Dataset # 2")
make_classify(make_classification, classifiers, names)
