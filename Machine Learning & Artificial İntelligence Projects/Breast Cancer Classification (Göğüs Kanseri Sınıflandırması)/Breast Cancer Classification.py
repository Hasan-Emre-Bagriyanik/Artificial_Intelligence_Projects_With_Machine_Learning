# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:53:23 2023

@author: Hasan Emre
"""

#%% import Library

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

# Warning library
import warnings
warnings.filterwarnings("ignore")

#%% 
# We are reading our csv file (csv dosyamizi okuyoruz)
data = pd.read_csv("data.csv")

# We remove unnecessary columns from our file (gereksiz sutunları dosyamizdan kaldiriyoruz) 
data.drop(["Unnamed: 32","id"], inplace=True, axis = 1)

# We changed the name of one of our columns (sutunlarimizdan birinin adini degistirdik)
data = data.rename(columns= {"diagnosis":"target"})

#%% 
# Visualization and number of malignant and benign in the target column (hedef sutununda kotu huylu ve iyi huylularin goruntulenmesi ve sayisi)
sns.countplot(data["target"])
print(data.target.value_counts())  # B (benign) = 357   M (malignant) = 212

#%% 
# We changed the two kinds of classes in the target column to 0 and 1  (Hedef sutundaki iki tur sinifi 0 ve 1 olarak degistirdik)
data["target"] =[1 if i.strip() == "M" else 0 for i in data.target]

#%%
# Data length  (Veri uzunlugu)
print(len(data))

# First 5 lines of data  ( Verinin ilk 5 satiri)
print(data.head())

# Data shape  (Veri sekli)
print("Data shape: ", data.shape)  # Data shape: (569, 31)

# Information about data  ( Veri hakkinda bilgi)
print(data.info())

# data describe (veri özeti)
describe = data.describe()
print("Describe: ", describe)

#%% standardization

"""
standardization 
missing value: none
"""


#%%
#%%
#%%   EDA
#%%

#%%

# Correlation 
# In the correlation map, a map was created to see if the columns themselves are directly proportional to each other
#(Korelasyon haritasinda, sutunlarin kendilerinin birbirleriyle dogru orantili olup olmadıgını gormek icin bir harita olusturulmustur.)
corr_matrix = data.corr()
sns.clustermap(corr_matrix,annot=True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()

#%%

# Since the correlation map is very complex , giving a threshold value here results in higher directly proportional columns
# (Korelasyon haritasi cok karmasik oldugu icin, burada bir esik degeri vermek, daha yuksek dogrudan orantili sutunlarla sonuclanir.)
threshold = 0.75
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot=True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Threshold 0.75")
plt.show()

#%%

# box plot
# We drew a graph with box plot, but the values turned out to be meaningless. we will standardize for this
# (Kutu cizimi ile bir grafik cizdik, ancak degerlerin anlamsiz oldugu ortaya cikti. Bunun icin standardize edecegiz)
data_melted = pd.melt(data,id_vars="target",
                      var_name="features",
                      value_name="value")
plt.figure()
sns.boxplot(x = "features", y = "value", hue="target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()

#%%
"""
standardization - normalization
"""

# pair plot
sns.pairplot(data[corr_features], diag_kind="kde", markers="+", hue="target")
plt.show()

"""
skewness
"""
#%%  Outlier

# The target column of data "target" is separated and assigned as "y" (Verilerin hedef sutunu olan 'target' ayrilir ve 'y' olarak atanir.)
y = data.target
# Columns other than the target column of the data are taken and assigned as 'x'.(Verinin hedef sutunu disindaki sutunlar alinir ve 'x' olarak atanir.)
x = data.drop(["target"], axis =1)
# Column names are assigned to the 'columns' list. (Sutun adlari 'columns' listesine atanir.)
columns = x.columns.tolist()


# A classifier (clf) is created from the LocalOutlierFactor class. (LocalOutlierFactor sinifindan bir siniflandirici (clf) olusturulur.)
clf = LocalOutlierFactor()
# It detects outliers using the LocalOutlierFactor classifier and takes the predictions as 'y_pred'.
# (LocalOutlierFactor siniflandiricisi kullanilarak aykiri degerleri tespit eder ve tahminleri 'y_pred' olarak alir.)
y_pred = clf.fit_predict(x)
# Outlier scores are taken for each observation and assigned as 'X_score'. (Her bir gözlem için aykırılık skorlari alinir ve 'X_score' olarak atanir.)
X_score = clf.negative_outlier_factor_

# A DataFrame containing the outlier scores is created. (Aykirilik skorlarini iceren bir DataFrame olusturulur.)
outlier_score = pd.DataFrame()
outlier_score["score"] = X_score


# The threshold value to be used to detect outliers is determined. (Aykiri degerlerin tespiti icin kullanilacak esik degeri belirlenir.)
threshold = -2.5
# To filter outliers, scores below the threshold are filtered. (# Aykiri degerleri filtrelemek icin esik degeri altinda olan skorlari filtreleme islemi yapilir.)
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()


plt.figure()
# Outlier blue values show colors as dots. (Aykiri degerleri mavi renkte nokta olarak gosterir.)
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1], color = "blue", s = 50, label = "Outlier")
# Shows normal data points as dots in black. (Normal veri noktalarini siyah renkte nokta olarak gosterir.)
plt.scatter(x.iloc[:,0], x.iloc[:,1], color ="k", s = 3, label = "Data Points")


# The radius is calculated based on the scores of the outliers and is circled in red.
# ( Aykiri degerlerin skorlarina gore yaricaplar hesaplanir ve kirmizi renkte daire icerisine alinarak gosterilir.)
radius = (X_score.max() - X_score)/ (X_score.max() - X_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0], x.iloc[:,1], s = 1000*radius, edgecolors="r", facecolors = "none", label  ="outlier Score")
plt.legend()
plt.show()


# drop outliers
# # Extract rows corresponding to outliers from dataset 'x'. ('x' veri kumesinden aykiri degerlere karsilik gelen satirlar cikarilir.)
x = x.drop(outlier_index)

# 'y' veri kümesinden aykırı değerlere karşılık gelen etiketler (hedef) çıkarılır ve 'y' olarak atanır.
# ('y' veri kumesinden aykiri degerlere karsilik gelen etiketler (hedef) cikarilir ve 'y' olarak atanir.)
y = y.drop(outlier_index).values


#%% train test split 
# used to separate the dataset into training and testing subsets.  (veri kumesini egitim ve test alt kumelerine ayirmak icin kullanilir)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
 
#%% standardation

# The StandardScaler class is used to scale the arguments (x_train and x_test) in the dataset. (Veri kumesindeki bagimsiz degiskenleri (x_train ve x_test) olceklendirme islemi icin StandardScaler sinifi kullanilir.)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# The standardized x_train dataset is converted to a DataFrame (x_train_df) with columns and column names (columns) are assigned to this DataFrame.
# (Standardize edilmis x_train veri kumesi, sutunları olan bir DataFrame'e (x_train_df) donusturulur ve sutun adları (columns) bu DataFrame'e atanir.)
x_train_df = pd.DataFrame(x_train, columns = columns)
# Extracts statistical summary of x_train_df  (x_train_df'in istatistiksel ozetine)
x_train_df_describe = x_train_df.describe()
# Add target variable (y_train) to dataset x_train_df  ( x_train_df veri kumesine hedef degiskeni (y_train) eklenir)
x_train_df["target"] = y_train


# Box plot
# The x_train_df dataset is visualized using a box plot. The box plot visually displays the distribution, median, and outliers for each feature.
# (x_train_df veri kumesi, kutu grafigi (box plot) kullanilarak gorsellestirilir. Box plot, her bir ozelligin dagilimini, ortanca degerini ve aykiri degerleri gorsel olarak gosterir.)
data_melted = pd.melt(x_train_df,id_vars="target",
                      var_name="features",
                      value_name="value")
plt.figure()
sns.boxplot(x = "features", y = "value", hue="target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()


# pair plot
# A pair plot (pair plot) is drawn to visually examine the relationships between pairs of some properties of the x_train_df dataset.
# (x_train_df veri kumesinin bazi ozelliklerinin ciftler arasindaki iliskileri gorsel olarak incelenmesi icin bir cift plot (pair plot) çizilir.)
sns.pairplot(x_train_df[corr_features], diag_kind="kde", markers="+", hue="target")
plt.show()



#%%
#%%  KNN  (K-Nearest Neighbors)
#%%
#%% Basic KNN Method

# The code trains the k-nearest neighbor (KNN) classifier with the training data, predicts the test data, evaluates the accuracy of the predictions, and prints the results.
# (Kod, k-en yakin komsu (KNN) siniflandiircisini egitim verileriyle egitiyor, test verilerini tahmin ediyor, tahminlerin dogrulugunu degerlendiriyor ve sonuclari yazdiriyor.)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
score = knn.score(x_test, y_test)
print("Score: ", score)
print("CM: ", cm)
print("Basic KNN accuracy: ", acc)

#   Output

# Score:  0.9532163742690059
# CM:  [[108   1]
#  [  7  55]]
# Basic KNN accuracy:  0.9532163742690059



#%% Choose best parameters

def KNN_best_params(x_train, x_test, y_train, y_test):
    # For the k-nearest neighbor (KNN) classifier, it finds the best k-value (from 1 to 30) and weight options ("uniform" and "distance") 
    # using GridSearchCV and trains it with training data to display the best score and best parameters. prints.
    # (k-en yakin komsu (KNN) siniflandiricisi icin, GridSearchCV'yi kullanarak en iyi k-degerini (1'den 30'a kadar) ve agirlik 
    # seceneklerini ("uniform" ve "mesafe") bulur ve en iyi skoru goruntulemek için egitim verileriyle egitir ve en iyi parametreler.) baskılar.
    k_range = list(range(1,31))
    weight_options = ["uniform", "distance"]
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring="accuracy")
    grid.fit(x_train, y_train)
    
    print("Best training score: {} with parameters: {}".format(grid.best_score_,grid.best_params_))
    print()
    
    #  Output
    # Best training score: 0.9670512820512821 with parameters: {'n_neighbors': 4, 'weights': 'uniform'}
    
    
    # It builds and trains a KNN classifier with the best parameters found with GridSearchCV, then makes predictions on test and 
    # training data and prints accuracy scores and confusion matrices.
    # (GridSearchCV ile bulunan en iyi parametrelerle bir KNN siniflandiricisi olusturup egitir, ardindan test ve egitim verileri 
    # uzerinde tahmin yaparak dogruluk skorlarini ve karisiklik matrislerini ekrana yazdirir.)

    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train,y_train)

    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    print("Test Score: {}, Train Score: {}".format(acc_test,acc_train))
    print()
    print("CM Test: ",cm_test)
    print("CM Train: ", cm_train)
    
    return grid

    # Output 
    # Test Score: 0.9590643274853801, Train Score: 0.9773299748110831

    # CM Test:  [[107   2]
    #  [  5  57]]
    # CM Train:  [[248   0]
    #  [  9 140]]



grid = KNN_best_params(x_train, x_test, y_train, y_test)


#%%  
#%%  PCA  (Principal Component Analysis)
#%%
#%% 
# It simplifies data analysis by representing the dataset on a two-dimensional plane with PCA and creating a scatterplot 
# with the target classes by color.
# (Veri setini PCA ile iki boyutlu bir duzlemde temsil ederek ve renge gore hedef siniflarla bir dagilim grafigi olusturarak veri analizini basitlestirir.)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components= 2)
pca.fit(x_scaled)
x_reduced_pca = pca.transform(x_scaled)

pca_data = pd.DataFrame(x_reduced_pca, columns = ["p1","p2"])
pca_data["target"] = y
sns.scatterplot(x = "p1", y  = "p2", hue="target", data = pca_data)
plt.title("PCA: p1 vs p2")

# It visualizes the PCA-reduced dataset on a two-dimensional plane using the K-Nearest Neighbor (KNN) classifier and 
# shows how the different classes are distributed in the colored regions.
# (PCA ile boyut azaltılmış veri kümesini K-En Yakın Komşu (KNN) sınıflandırıcısı kullanarak iki boyutlu bir düzlemde görselleştirir
# ve farklı sınıfların renkli bölgelerde nasıl dağıldığını gösterir.)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x_reduced_pca, y, test_size=0.3, random_state=42)

grid_pca = KNN_best_params(x_train_pca, x_test_pca, y_train_pca, y_test_pca)


# visualize  (Gorsellestirme)
cmap_light = ListedColormap(["orange", "cornflowerblue"])
cmap_bold = ListedColormap(["darkorange", "darkblue"])

h = .05 # step size in the mesh   (Agdaki adim boyutu)
x = x_reduced_pca

x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max,h ),
                     np.arange(y_min, y_max, h))

z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot   (Sonucu renkli bir cizime koyun)
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, z, cmap=cmap_light)

# Plot also the training points  (Egitim noktalarini da cizin)
plt.scatter(x[:,0], x[:, 1], c = y, cmap=cmap_bold,
            edgecolors="k", s = 20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
    % (len(np.unique(y)), grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights))

# Output 

# Test Score: 0.9239766081871345, Train Score: 0.947103274559194

# CM Test:  [[103   6]
#  [  7  55]]
# CM Train:  [[241   7]
#  [ 14 135]]
# Out[107]: Text(0.5, 1.0, "2-Class classification (k = 9, weights = 'uniform')")


#%% 
#%%  NCA (Neighborhood Components Analysis)
#%%
#%%
# Using Neighborhood Components Analysis (NCA), it visualizes the downsized dataset on a two-dimensional plane and 
# shows how the different target classes are dispersed in non-color.
# ( Komsu Bilesenler Analizi (Neighborhood Components Analysis - NCA) yontemini kullanarak boyut azaltilmis veri kumesini iki boyutlu
#  bir duzlemde gorsellestirir ve farkli hedef siniflarin renkli bolgelerde nasil dağilimini gosterir.)

nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
nca.fit(x_scaled, y)
x_reduced_nca = nca.transform(x_scaled)
nca_data  =pd.DataFrame(x_reduced_nca, columns = ["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x = "p1", y="p2", hue="target", data = nca_data)
plt.title("NCA: p1 vs p2")



x_train_nca, x_test_nca, y_train_nca, y_test_nca = train_test_split(x_reduced_nca, y, test_size=0.3, random_state=42)

grid_nca = KNN_best_params(x_train_nca, x_test_nca, y_train_nca, y_test_nca)



# visualize  (Gorsellestirme)
cmap_light = ListedColormap(["orange", "cornflowerblue"])
cmap_bold = ListedColormap(["darkorange", "darkblue"])

h = 2 # step size in the mesh   (Agdaki adim boyutu)
x = x_reduced_nca

x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max,h ),
                     np.arange(y_min, y_max, h))

z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot   (Sonucu renkli bir cizime koyun)
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, z, cmap=cmap_light)

# Plot also the training points  (Egitim noktalarini da cizin)
plt.scatter(x[:,0], x[:, 1], c = y, cmap=cmap_bold,
            edgecolors="k", s = 20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
    % (len(np.unique(y)), grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))


# Output
# Test Score: 0.9941520467836257, Train Score: 1.0

# CM Test:  [[108   1]
#  [  0  62]]
# CM Train:  [[248   0]
#  [  0 149]]


