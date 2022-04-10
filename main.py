import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('data.tsv', sep='\t')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
data.interpolate(method ='linear', limit_direction ='forward')
pd.DataFrame(data).fillna(data.mean(), inplace=True)
data.interpolate()


#visualize
datagraph = pd.read_csv('dataoriginal.tsv', sep='\t')
sns.set(style="darkgrid")


#diagram 1
show=sns.FacetGrid(data,col='Overall Survival Status')
show.map(plt.hist,'Diagnosis Age')

#diagram 2
show=sns.FacetGrid(data,col='Sex')
show.map(plt.hist,'Overall Survival Status')

plt.show()

train=data.drop([
    'Study ID',
'Patient ID',
'Sample ID',
# 'Diagnosis Age',
'Neoplasm Disease Stage American Joint Committee on Cancer Code',
'American Joint Committee on Cancer Publication Version Type',
# 'Aneuploidy Score',
# 'Buffa Hypoxia Score',
'Cancer Type',
'TCGA PanCanAtlas Cancer Type Acronym',
'Cancer Type Detailed',
'Last Communication Contact from Initial Pathologic Diagnosis Date',
#'Birth from Initial Pathologic Diagnosis Date',
#'Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value',
 'Disease Free (Months)',
 #'Disease Free Status',
 #'Months of disease-specific survival',
#'Disease-specific Survival status',
'Ethnicity Category',
'Form completion date',
#'Fraction Genome Altered',
'Neoplasm Histologic Grade',
'Neoadjuvant Therapy Type Administered Prior To Resection Text',
'ICD-10 Classification',
'International Classification of Diseases for Oncology, Third Edition ICD-O-3 Histology Code',
'International Classification of Diseases for Oncology, Third Edition ICD-O-3 Site Code',
'Informed consent verified',
'In PanCan Pathway Analysis',
#'MSI MANTIS Score',
#'MSIsensor Score',
#'Mutation Count',
'New Neoplasm Event Post Initial Therapy Indicator',
'Oncotree Code',	
#'Overall Survival (Months)',
'Overall Survival Status',
'Other Patient ID',
'American Joint Committee on Cancer Metastasis Stage Code',
'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code',
'American Joint Committee on Cancer Tumor Stage Code',
'Person Neoplasm Cancer Status',
#'Progress Free Survival (Months)',	
'Progression Free Status',	
'Primary Lymph Node Presentation Assessment',
'Prior Diagnosis',
'Race Category',
'Radiation Therapy',
#'Ragnum Hypoxia Score',
'Number of Samples Per Patient',
'Sample Type',
'Sex',
'Somatic Status',
'Subtype',
'Tissue Prospective Collection Indicator',
'Tissue Retrospective Collection Indicator',
'Tissue Source Site',
'Tissue Source Site Code',
#'TMB (nonsynonymous)',	
'Tumor Disease Anatomic Site',
'Tumor Type'
#'Patient Weight',
#'Winter Hypoxia Score'
 ],axis=1)


test=data['Overall Survival Status']


X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3,random_state=2)

regr = LinearRegression()

regr.fit(X_train,y_train)
pred = regr.predict(X_test)
print(pred)
print("Accuracy =",round(regr.score(X_test,y_test)*100,2),"%")



