#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from scipy import stats
from tqdm import tqdm


# In[2]:


inputFile = "MAESTRO-d6178bdd-identified_variants_merged_protein_regions-main.tsv"
data = pd.read_csv(inputFile,sep='\t',low_memory=False)


# In[3]:


data_processed = data[['Peptide'] + [c for c in data.columns if 'intensity_for_peptide_variant' in c]]


# In[4]:


data_processed.replace(0.0,np.nan, inplace = True)


# In[5]:


data_processed = data_processed.set_index("Peptide")


# In[6]:


data_processed = data_processed.T


# In[7]:


data_processed.index = data_processed.index.map(lambda x:'.'.join(x.split('.')[:2]))


# In[8]:


def create_label(x):
    if "#Healthy" in x:
        return 1
    elif "#Non-severe-COVID-19" in x:
        return 2
    elif "#Severe-COVID-19" in x:
        return 3
    elif "#Symptomatic-non-COVID-19" in x:
        return 4
    else:
        return 0
data_processed["label"]=data_processed.index.map(lambda x: create_label(x))


# In[9]:


data_processed = data_processed[data_processed['label']!=0]


# In[10]:


#Split Train and Test data after shuffling 
data_processed = data_processed.sample(frac=1, random_state=42)
data_processed.dropna(axis=1,inplace=True)

train = data_processed.iloc[:66,:]
test = data_processed.iloc[66:,:]


# In[ ]:





# In[11]:


# #Filter Data based on NaN for each label in training data

# per_label_count = train.groupby('label').count()
# filterCondition = (per_label_count >= 13).all()
# filter_per_label = per_label_count.loc[:, filterCondition]
# filter_column = list(filter_per_label.columns) +['label']
# filter_data =  train[filter_column]

# for column in tqdm(filter_data):
#     if column != "label":
#         filter_data[column] = filter_data.groupby("label")[column].transform(lambda x: x.fillna(x.mean()))

        
# ##filter_data.to_csv("filterd_data.tsv", sep="\t")
# filter_data.shape

## Normalize train data
# for column in tqdm(train):
#     if column != "label":
#         train[column] = train.groupby("label")[column].transform(lambda x: ((x - x.min()) / (x.max()-x.min())))


# In[12]:


##Ttest 
from itertools import combinations
from scipy import stats
from collections import defaultdict

grouped_df = train.groupby('label')
label_key = list(grouped_df.groups.keys())
peptide_key = list(grouped_df.get_group(label_key[0]).keys())


def ttest_run(c1, c2):
    results = stats.ttest_ind(c1, c2, equal_var=False, alternative='two-sided')
    if results.pvalue < 0.05:
        return 1
    return 0
    

imp = []
for p in tqdm(peptide_key):
    group_peptide = defaultdict(dict)
    for k in label_key: 
        group_peptide[k] = grouped_df.get_group(k)[p]
    
    ttest = [ttest_run(list(group_peptide[i]),list(group_peptide[j])) for i, j in combinations(label_key, 2)]
    if sum(ttest) > 0:
        imp.append(p)


# In[13]:


train = train[imp]
print(len(imp))


# In[14]:


#Correlation Matrix
corr_matrix = train.corr()
corr_matrix = corr_matrix.abs()
columns = np.full((corr_matrix.shape[0],), True, dtype=bool)
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[0]):
        if corr_matrix.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = train.columns[columns]
train = train[selected_columns]

print(len(selected_columns))
print(train.shape)


# In[15]:


test = test[train.columns]


# In[16]:


## Logistic Regression 
X_train, Y_train = train.iloc[:,:-1], train.iloc[:,-1]
X_test, Y_test = test.iloc[:,:-1], test.iloc[:,-1]


# In[17]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
from sklearn.metrics import  classification_report


# In[18]:


# # Logistic Regression
# lr = LogisticRegression(random_state=20)
# lr.fit(X_train,Y_train)
# lr_Y = lr.predict(X_test)
# lr_acc = accuracy_score(Y_test, lr_Y)
# print(lr_acc)
# print(classification_report(Y_test, lr_Y))


# In[19]:


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# # svm_classifier = SVC(kernel="linear")
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
# classifier.fit(X_train, Y_train)
# y_pred = classifier.predict(X_test)
# rf_acc = accuracy_score(Y_test, y_pred)
# rf_acc


# In[20]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


# In[21]:


lr = LogisticRegression(random_state=20)


# In[22]:


min_features_to_select = 800  # Minimum number of features to consider
rfecv_lr = RFECV(
    estimator=lr,
    step=5,
    cv=StratifiedKFold(5),
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
)


# In[23]:


rfecv_lr.fit(X_train, Y_train)
rfecv_lr_Y = rfecv_lr.predict(X_test)
rfecv_lr_acc = accuracy_score(Y_test, rfecv_lr_Y)


# In[24]:


print(classification_report(Y_test, rfecv_lr_Y))


# In[25]:


final_feature = X_train[X_train.columns[rfecv_lr.ranking_ == 1]]
feature_X = list(final_feature.columns)
print(len(feature_X))


# In[26]:


from sklearn.feature_selection import SelectKBest, chi2

X_Kbest = SelectKBest(chi2, k=20).fit(final_feature, Y_train)
cols = X_Kbest.get_support(indices=True)
feature_X_Kbest= final_feature.iloc[:,cols]


# In[27]:


feature_Kbest = list(feature_X_Kbest.columns)


# In[28]:


protien_data = data[['Peptide']+["Canonical_proteins"]] #Canonical_proteins
protien_data = protien_data.set_index("Peptide")
protien_data = protien_data[protien_data.index.isin(feature_Kbest)]
protien_data


# In[29]:


def getAllProteins(proteins):
    proteinDict = {}
    proteinList = []
    listOfProtein =proteins.split(";") 
    for protein in listOfProtein:
        data = protein.split("|")
        if len(data)<2:
            continue
        proteinList.append(data[1])
        proteinDict[data[1]] = protein
    return proteinList, proteinDict


# In[30]:


from collections import defaultdict

def getPeptideAndProtein(protien_data):
    
    peptideToProteins = defaultdict(list)
    proteinToPeptides = defaultdict(list)

    ProteinsDict = {}

    for index, row in tqdm(protien_data.iterrows()):
        proteins =str(row["Canonical_proteins"]) 
        proteinList, proteinDict = getAllProteins(proteins)
        if len(proteinList)==0:
            continue
        ProteinsDict.update(proteinDict)
        peptideToProteins[index].extend(proteinList)
        for p in proteinList:
            proteinToPeptides[p].append(index)
    return peptideToProteins, proteinToPeptides, ProteinsDict


# In[31]:


peptideToProteins, proteinToPeptides, ProteinsDict = getPeptideAndProtein(protien_data)


# In[32]:


uniqueProtein = set()

for protein in proteinToPeptides:
    if len(proteinToPeptides[protein])==1 and len(peptideToProteins[proteinToPeptides[protein][0]])==1:
        uniqueProtein.add(protein)
print(len(uniqueProtein))


# In[33]:


len(peptideToProteins)


# In[34]:


len(proteinToPeptides)


# In[35]:


print(uniqueProtein)


# In[36]:


includedPeptide = {}

for peptide in peptideToProteins:
    includedPeptide[peptide] = False
    
for protein in uniqueProtein:
    includedPeptide[proteinToPeptides[protein][0]]=True


# In[37]:


proteinToPeptidesSortedList =  sorted(proteinToPeptides, key = lambda key: -len(proteinToPeptides[key]))


# In[38]:


len(includedPeptide)


# In[39]:


print(proteinToPeptidesSortedList)


# In[40]:


def countMaxPeptideCount(includedPeptide, uniqueProtein):
    maxLen = 0
    maxProtein = ""
    for protein in proteinToPeptidesSortedList:
        if protein in uniqueProtein:
            continue
                
        peptideList = proteinToPeptides[protein]
        count=0
        
        for p in peptideList:
            if includedPeptide[p]==False:
                count+=1
        if count > maxLen:
            maxLen = count 
            maxProtein = protein
    return maxProtein, maxLen


# In[41]:


Total = len(includedPeptide)

while Total != sum(includedPeptide.values()):
        protein, length = countMaxPeptideCount(includedPeptide, uniqueProtein)
        peptideList = proteinToPeptides[protein]
        for peptide in peptideList:
            includedPeptide[peptide] = True
        uniqueProtein.add(protein)
        print(protein, length, len(proteinToPeptides[protein]))
        


# In[60]:


topProtein = {} 
for i in uniqueProtein:
    topProtein[i]=proteinToPeptides[i]
    
topProteinList = sorted(topProtein, key = lambda key: -len(topProtein[key]))#[:10]


# In[61]:


topProteinList


# In[44]:


all_protien_data = data[['Peptide']+["Canonical_proteins"]]
all_protien_data = all_protien_data.set_index("Peptide")


# In[45]:


allPeptideToProteins, allProteinToPeptides, allProteinsDict = getPeptideAndProtein(all_protien_data)


# In[46]:


Tcount = len(feature_X)

for tpl in topProteinList:
    count = 0
    allPeptide = allProteinToPeptides[tpl]
    for p in feature_X:
        if p in allPeptide:
            count+=1
    print(tpl, " : ",count ," : ",(count/Tcount))
                


# In[53]:


from networkx.algorithms import bipartite
import networkx as nx


# In[52]:


topPeptideList = list(peptideToProteins.keys())


# In[59]:


topProteinList


# In[54]:


G = nx.Graph()
G.add_nodes_from(topProteinList, bipartite=0)
G.add_nodes_from(topPeptideList,bipartite=1)


# In[62]:


edge = []
for i in topProtein:
    for j in topProtein[i]:
        edge.append((i,j))
        


# In[64]:


G.add_edges_from(edge)


# In[91]:


nx.draw_networkx(G, pos = nx.drawing.layout.bipartite_layout(G,topProteinList), width = 1,node_size=500, node_color ="blue", alpha=0.65)


# In[ ]:




