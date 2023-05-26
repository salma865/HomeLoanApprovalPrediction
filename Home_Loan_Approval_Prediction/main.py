import statistics
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('loan_sanction_train.csv')

# ############################ Preprocessing ############################

# Drop Loan_ID column
df_train.drop('Loan_ID', axis=1, inplace=True)

# check missing value
print(df_train.isna().sum())

# fill missing data with mode and mean
for column in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    df_train[column].fillna(df_train[column].mode()[0], inplace=True)
for column in ['LoanAmount', 'Loan_Amount_Term']:
    df_train[column].fillna(df_train[column].mean(), inplace=True)


# check missing again
print(df_train.isna().sum())
# check duplicate
print(df_train.duplicated())
# check again duplicate
df_train.drop_duplicates(inplace=True)
print(df_train.duplicated())

# category encoding
df_train.dtypes
print(df_train['Gender'].unique())
print(df_train['Married'].unique())
print(df_train['Dependents'].unique())
print(df_train['Self_Employed'].unique())
print(df_train['Education'].unique())
print(df_train['Property_Area'].unique())
print(df_train['Loan_Status'].unique())
df_train=df_train.replace({
    'Gender': {'Male': 0, 'Female': 1},
    'Married': {'No': 0, 'Yes': 1},
    'Dependents': {'3+': 3},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Property_Area': {'Urban': 0, 'Rural': 1, 'Semiurban': 2},
    'Loan_Status': {'Y': 1, 'N': 0},
})
df_train.head()

# feature scaling
# normalization on training data
scaler = MinMaxScaler()
columns_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler.fit(df_train[columns_to_scale])
scaled_data = scaler.transform(df_train[columns_to_scale])
scaled_data = pd.DataFrame(scaled_data, columns=columns_to_scale)
scaled_data = pd.concat([df_train.drop(columns_to_scale, axis=1), scaled_data], axis=1)
print(scaled_data.head())
print(scaled_data.shape)

# ############################ Detect Outliers ############################

# define a function called "plot_boxplot"


def plot_boxplot(df, ft):
    df.boxplot(column=[ft])
    plt.grid(True)
    plt.show()


# show boxplot before remove outliears
plot_boxplot(scaled_data, "LoanAmount")
plot_boxplot(scaled_data, "Loan_Amount_Term")
plot_boxplot(scaled_data, "CoapplicantIncome")
plot_boxplot(scaled_data, "ApplicantIncome")

# remove outliers


def outliers(df, ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3-Q1

    lower_bound = Q1-1.5*IQR
    upper_bound = Q3 + 1.5 * IQR
    ls = df.index[(df[ft] < lower_bound) | (df[ft] > upper_bound)]
    return ls

# creat an empty list to store the output indices from multiple coulmns


index_list = []
for feature in ['LoanAmount', 'Loan_Amount_Term', 'CoapplicantIncome', 'ApplicantIncome']:
    index_list.extend(outliers(scaled_data, feature))

print(index_list)

# define function called "remove" which returns cleaned dataframe without outliers


def remove(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df


df_train_cleaned = remove(scaled_data, index_list)
print(df_train_cleaned.shape)
# show boxplot after remove outliers
plot_boxplot(df_train_cleaned, "LoanAmount")
plot_boxplot(df_train_cleaned, "Loan_Amount_Term")
plot_boxplot(df_train_cleaned, "CoapplicantIncome")
plot_boxplot(df_train_cleaned, "ApplicantIncome")
print(df_train_cleaned.head())

# ############################ Statistics ############################

# 1) for Numerical attributes: use describe function
print(df_train.describe())
print("ApplicantIncome column measures:")
print("Median: ", np.median(df_train['ApplicantIncome']),
      "Mode: ", statistics.mode(df_train['ApplicantIncome']),
      "Variance: ", np.var(df_train['ApplicantIncome']))
print("CoapplicantIncome column measures:")
print("Median: ", np.median(df_train['CoapplicantIncome']),
      "Mode: ", statistics.mode(df_train['CoapplicantIncome']),
      "Variance: ", np.var(df_train['CoapplicantIncome']))
print("LoanAmount column measures:")
print("Median: ", np.median(df_train['LoanAmount']),
      "Mode: ", statistics.mode(df_train['LoanAmount']),
      "Variance: ", np.var(df_train['LoanAmount']))

# 2) for categorical attributes : use value counts function
print(df_train['Gender'].value_counts())
print(df_train['Married'].value_counts())
print(df_train['Loan_Amount_Term'].value_counts())
print(df_train['Self_Employed'].value_counts())
print(df_train['Education'].value_counts())
print(df_train['Credit_History'].value_counts())
print(df_train['Loan_Status'].value_counts())
print(df_train['Property_Area'].value_counts())

# ############################ Visualization ############################

# 1) Visualize distribution of data for ApplicantIncome and LoanAmount columns using histogram
sea.histplot(data=df_train_cleaned, x='ApplicantIncome', color='red', kde=True, bins=30)
plt.show()
sea.histplot(data=df_train_cleaned, x='LoanAmount', color='blue', kde=True, bins=30)
plt.show()

# 2) Visualize some of categorical data using bie chart
df_train_cleaned.groupby('Married').size().plot(kind='pie', autopct='%.2f')
plt.show()
df_train_cleaned.groupby('Credit_History').size().plot(kind='pie', autopct='%.2f')
plt.show()

# 3) Study Impact of Gender, Education, Self Employed and Credit History on Loan Status using count plot
sea.countplot(data=df_train_cleaned, x="Gender", hue="Loan_Status")
plt.title('Gender / Loan_Status')
plt.show()

sea.countplot(data=df_train_cleaned, x="Education", hue="Loan_Status")
plt.title('Education / Loan_Status')
plt.show()

sea.countplot(data=df_train_cleaned, x="Self_Employed", hue="Loan_Status")
plt.title('Self_Employed / Loan_Status')
plt.show()

sea.countplot(data=df_train_cleaned, x="Credit_History", hue="Loan_Status")
plt.title('Credit_History / Loan_Status')
plt.show()

# 4) Study Impact of Property_Area on Loan Status using count plot
sea.countplot(data=df_train_cleaned, x="Property_Area", hue="Loan_Status")
plt.title('Property_Area / Loan_Status')
plt.show()

# 5) Show which area applicant choose according to his income using density plot
sea.displot(data=df_train_cleaned, x="ApplicantIncome", hue="Property_Area", kind="kde")

# 6) Impact of Income on Approved LoanAmount
plt.scatter(x=df_train_cleaned['LoanAmount'], y=df_train_cleaned['ApplicantIncome'], c=df_train_cleaned['Property_Area']
            , cmap="tab20")
plt.legend(('Urban', 'Rural', 'Semiurban'))
plt.xlabel("LoanAmount")
plt.ylabel("ApplicantIncome")
plt.show()

# 7) Study relationships between attributes using correlation, visualize using heat map
plt.figure(figsize=(10, 5))
sea.heatmap(data=df_train_cleaned.corr(method='spearman'), annot=True)
plt.show()

# ############################ Classification ############################

# Splitting data into train and test
X = df_train_cleaned.drop('Loan_Status', axis=1)
Y = df_train_cleaned['Loan_Status']
XTrain, Xtest, YTrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

# KNN                         ------>> accuracy 73.23%
KNN_model = neighbors.KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(XTrain, YTrain)
KNN_predict = KNN_model.predict(Xtest)
print("KNN score:")
print(accuracy_score(Ytest, KNN_predict)*100)

# Decision Tree               ------>> accuracy 68.5%
DT_model = DecisionTreeClassifier()
DT_model.fit(XTrain, YTrain)
DT_y_pred = DT_model.predict(Xtest)
print("Decision tree model")
print(accuracy_score(Ytest, DT_y_pred)*100)
print("*******************")

# Random Forest Classifier    ------>> accuracy 76.05%     --->> Best classifier
RF_model = RandomForestClassifier()
RF_model.fit(XTrain, YTrain)
RF_y_pred = RF_model.predict(Xtest)
print("Random Forest model")
print(accuracy_score(Ytest, RF_y_pred)*100)
print("*******************")

# Naive Bayes classifier      ------>> accuracy 69.9%
NB_model = MultinomialNB()
NB_model.fit(XTrain, YTrain)
NB_y_pred3 = NB_model.predict(Xtest)
print("Naive Bayes model")
print(accuracy_score(Ytest, NB_y_pred3)*100)










