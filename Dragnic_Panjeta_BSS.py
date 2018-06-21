# Dragnic Z, Panjeta V.
# Arrythmia classification using ECG data

# Required Packages
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Import dataset
#data1 = np.genfromtxt('C:/Users/Dragnic/PycharmProjects/BSS/arrhythmia.data', dtype= None)
data1 = pd.read_csv('C:/Users/dana/Downloads/arrhythmia.data', header=None)
data2 = data1
data1 = data1.replace('?', 0)
data = data1.values
red,kolona = data.shape
target = data[:,279]
features = data[:,1:279]
print('Dataset has {0} rows and {1} columns'.format(*data.shape))

# Exploratory phase
# Histogram - arrythmia types

vektor = np.zeros(16)
for i in range (0,red-1):
    vektor[data[i][kolona-1] - 1] = vektor[data[i][kolona-1]-1] +1

plt.figure(figsize=(5,5))
x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
y1 = vektor
plt.bar(x1,y1,align='center') # A bar chart
plt.xlabel('Tip aritmije')
plt.ylabel('Frekvencija')
plt.title('Histogram - tip aritmije')
plt.show()

# Pie chart - years
brojPogodinama = np.zeros(7)
tipAritmije = np.zeros((7,16))
#15-25; 25 -35; 35-45; 35-55;55-65;65-75;75-85
for i in range(0,red-1):
    if(data[i][0]>=15 and data[i][0]<25):
        brojPogodinama[0]+=1
        tip = data[i][kolona-1]
        tipAritmije[0][tip-1] += 1
    elif(data[i][0]>=25 and data[i][0]<35):
        brojPogodinama[1]+=1
        tip = data[i][kolona-1]
        tipAritmije[1][tip-1] += 1
    elif (data[i][0] >= 35 and data[i][0] < 45):
        brojPogodinama[2] += 1
        tip = data[i][kolona-1]
        tipAritmije[2][tip-1] += 1
    elif (data[i][0] >= 45 and data[i][0] < 55):
        brojPogodinama[3] += 1
        tip = data[i][kolona-1]
        tipAritmije[3][tip-1] += 1
    elif (data[i][0] >= 55 and data[i][0] < 65):
        brojPogodinama[4] += 1
        tip = data[i][kolona-1]
        tipAritmije[4][tip-1] += 1
    elif (data[i][0] >= 65 and data[i][0] < 75):
        brojPogodinama[5] += 1
        tip = data[i][kolona-1]
        tipAritmije[5][tip-1] += 1
    elif (data[i][0] >= 75 and data[i][0] < 85):
        brojPogodinama[6] += 1
        tip = data[i][kolona-1]
        tipAritmije[6][tip-1] += 1

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

figureObject, axesObject = plt.subplots()
axesObject.pie(brojPogodinama,None,["15-25","25-35","35-45","45-55","55-65","65-75","75-85"],autopct=make_autopct(brojPogodinama))
axesObject.axis('equal')
plt.title('Starosna struktura')
plt.show()

# Pie chart - gender
spolVektor = np.zeros(2)
for i in range(0,red-1):
    if(data[i][1] == 1):
        spolVektor[1] += 1
    elif(data[i][1] == 0):
        spolVektor[0] += 1

figureObject, axesObject = plt.subplots()
axesObject.pie(spolVektor,None,["Musko", "Zensko"],autopct=make_autopct(spolVektor))
axesObject.axis('equal')
plt.title('Spolna struktura')
plt.show()

# Pie chart - weight
brojPoTezini = np.zeros(5)
for i in range(0,red-1):

    if(data[i][3]>=40 and data[i][3]<55):
        brojPoTezini[0]+=1
    elif(data[i][3]>=55 and data[i][3]<70):
        brojPoTezini[1]+=1
    elif (data[i][3] >= 70 and data[i][3] < 85):
        brojPoTezini[2] += 1
    elif (data[i][3] >= 85 and data[i][3] < 100):
        brojPoTezini[3] += 1
    elif (data[i][3] >= 100 and data[i][3] < 115):
        brojPoTezini[4] += 1


labels = ["40-55","55-70","70-85","85-100","100-115"]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#65cc99']
fig1, ax1 = plt.subplots()
ax1.pie(brojPoTezini, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax1.axis('equal')  
plt.tight_layout()
plt.title('Težina ispitanika [kg]')
plt.show()


# Scatter plot - Hearth rate and years
x1 = data[:,0]
y1 = data[:,14]
y1 = np.array(y1, dtype=float)
plt.plot(x1, y1, 'o')
plt.title('Heart Rate')
plt.ylim(20,140)
plt.xlim(10,100)
plt.xlabel('Godine')
plt.ylabel('HR [BPM]')
plt.show()

# Histogram - Hearth rate and gender (Average)
prosjekHR_spol=np.zeros(2)
prosjek_musko=0;
prosjek_zensko=0;
for i in range(0,red-1):

    if(data[i][1]==0):
        prosjek_musko = prosjek_musko + float(data[i][14])

    elif (data[i][1]==1):
        prosjek_zensko = prosjek_zensko + float(data[i][14])


prosjek_musko= prosjek_musko/spolVektor[0]
print("prosjek HR muskarci:",prosjek_musko)
prosjek_zensko = prosjek_zensko/spolVektor[1]
print("prosjek HR zene:",prosjek_zensko)
prosjekHR_spol[0]=prosjek_musko
prosjekHR_spol[1]=prosjek_zensko

plt.figure(figsize=(5,5))
colorHR = ['#66b3ff','#ff9999']
x1 = ['Muškarci','Žene']
y1 = prosjekHR_spol
plt.bar(x1,y1,align='center',color=colorHR) # A bar chart
plt.xlabel('Tip aritmije')
plt.ylabel('Frekvencija')
plt.title('Prosječni HR u odnosu na spol')
plt.show()

# QRS duration and years
x1 = data[:,0]
y1 = data[:,4]
y1 = np.array(y1, dtype=float)
plt.plot(x1, y1, 'o')
plt.title('QRS trajanje')
plt.ylim(0,200)
plt.xlim(10,100)
plt.xlabel('Godine')
plt.ylabel('Vrijene [ms]')
plt.show()

# prosjecno trajanje QRS kompleksa u odnosu na tip aritmije
prosjekQRS=np.zeros(16)
brojIspitanika=np.zeros(16)
for i in range(0,red-1):
    if(target[i]==1):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==2):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==3):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==4):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==5):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==6):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==7):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==8):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==9):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==10):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==11):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==12):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==13):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==14):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==15):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1
    elif (target[i]==16):
        prosjekQRS[target[i]-1]=prosjekQRS[target[i]-1]+data[i][4]
        brojIspitanika[target[i]-1]=brojIspitanika[target[i]-1] + 1

for i in range(0,16):
    prosjekQRS[i]=prosjekQRS[i]/brojIspitanika[i]

plt.figure(figsize=(5,5))
x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
y1 = prosjekQRS
plt.bar(x1,y1,align='center') # A bar chart
plt.xlabel('Tip aritmije')
plt.ylabel('Prosječno trajanje QRS [ms]')
plt.title('QRS trajanje za tipove aritmije')
plt.show()

def fit_model(X_train, X_test, y_train, y_test):

    # Kreiranje objekta klasifikatora; random_state=0 znači da bi se pri svakom pokretanju trebali dobiti isti rezultati
    # criterion - entropy or gini
    # max_depth - levels of tree
    classificator = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=15,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')


    # Treniranje modela - trening se vrši isključivo nad trening podacima
    classificator.fit(X_train, y_train)

    # Određivanje koja osobina nosi najveću informacijsku dobiti (Gini index)
    print("Osobine (Gini importance): ", classificator.feature_importances_)

    # Skor za testne podatke - 1.0 (R^2 score)
    print("R2 score - trening podaci:{0}".format(classificator.score(X_train, y_train)))

    return classificator


features=features.astype('int')
target=target.astype('int')

# Training and test data - separation
X_train, X_test, y_train, y_test = train_test_split( features, target, test_size = 0.2, random_state = 100)
classificator = fit_model(X_train, X_test, y_train, y_test)
predictions=classificator.predict(X_test)

#R2 SCORE
print("R2 score - testni podaci:{0}".format(classificator.score(X_test, y_test)))

#ACCURACY SCORE
from sklearn.metrics import accuracy_score
print ("Accuracy is ", accuracy_score(y_test,predictions)*100)


# Export decision tree as png
from graphviz import Source
from IPython.display import SVG


Source( tree.export_graphviz(classificator, out_file=None, feature_names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','196','197','198','199','200','201','202','203','204','205','206','207','208','209','210','211','212','213','214','215','216','217','218','219','220','221','222','223','224','225','226','227','228','229','230','231','232','233','234','235','236','237','238','239','240','241','242','243','244','245','246','247','248','249','250','251','252','253','254','255','256','257','258','259','260','261','262','263','264','265','266','267','268','269','270','271','272','273','274','275','276','277','278']))
graph = Source(tree.export_graphviz(classificator, out_file=None, feature_names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','196','197','198','199','200','201','202','203','204','205','206','207','208','209','210','211','212','213','214','215','216','217','218','219','220','221','222','223','224','225','226','227','228','229','230','231','232','233','234','235','236','237','238','239','240','241','242','243','244','245','246','247','248','249','250','251','252','253','254','255','256','257','258','259','260','261','262','263','264','265','266','267','268','269','270','271','272','273','274','275','276','277','278']))
SVG(graph.pipe(format='svg'))

graph = Source( tree.export_graphviz(classificator, out_file=None, feature_names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','196','197','198','199','200','201','202','203','204','205','206','207','208','209','210','211','212','213','214','215','216','217','218','219','220','221','222','223','224','225','226','227','228','229','230','231','232','233','234','235','236','237','238','239','240','241','242','243','244','245','246','247','248','249','250','251','252','253','254','255','256','257','258','259','260','261','262','263','264','265','266','267','268','269','270','271','272','273','274','275','276','277','278']))
graph.format = 'png'
graph.render('dtree_render',view=True)

graph = Source( tree.export_graphviz(classificator, out_file=None, feature_names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','196','197','198','199','200','201','202','203','204','205','206','207','208','209','210','211','212','213','214','215','216','217','218','219','220','221','222','223','224','225','226','227','228','229','230','231','232','233','234','235','236','237','238','239','240','241','242','243','244','245','246','247','248','249','250','251','252','253','254','255','256','257','258','259','260','261','262','263','264','265','266','267','268','269','270','271','272','273','274','275','276','277','278']))
png_bytes = graph.pipe(format='png')
with open('dtree_pipe.png','wb') as f:
    f.write(png_bytes)

# Show image
from IPython.display import Image
Image(png_bytes)



