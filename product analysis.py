import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
try:
   df=pd.read_excel("D:\internship AK\product rfm analysis data.xlsx")
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found. Please ensure the file exists in the correct directory.")
    exit()
if not pd.api.types.is_datetime64_any_dtype(df['PurchaseDate']):
    try:
        df['PurchaseDate'] = pd.to_timedelta(df['PurchaseDate'], unit='d') + pd.Timestamp('1899-12-30')
    except:
        print("Error: PurchaseDate column is not in a recognizable date format.")
        exit()
else:
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
df['TransformationAmount']=pd.to_numeric(df['TransactionAmount'],errors='coerce')
df=df.dropna(subset=['CustomerID','PurchaseDate','TransactionAmount'])
analysis_date=df['PurchaseDate'].max()+timedelta(days=1)
rfm=df.groupby('CustomerID').agg({'PurchaseDate':lambda x:(analysis_date-x.max()).days,
                                  'OrderID':'count',
                                  'TransactionAmount':'sum'}).reset_index()
rfm.columns=['CustomerID','Recency','Frequency','Monetary']
rfm['R_Score']=pd.qcut(rfm['Recency'],5,labels=[5,4,3,2,1],duplicates='drop')
rfm['F_Score']=pd.qcut(rfm['Frequency'].rank(method='first'),5,labels=[1,2,3,4,5])
rfm['M_Score']=pd.qcut(rfm['Monetary'],5,labels=[1,2,3,4,5],duplicates='drop')
rfm['RFM_Score']=rfm['R_Score'].astype(str)+rfm['F_Score'].astype(str)+rfm['M_Score'].astype(str)
def segment_customer(row):
    if row['R_Score'] >=4 and row['F_Score'] >= 4 and row['M_Score'] >=4:
        return 'High-Value'
    elif row['R_Score'] >=3 and row['F_Score'] >=3:
        return 'Loyal'
    elif row['R_Score'] >=3 and row['M_Score'] >=3:
        return 'Big Spender'
    elif row['R_Score'] <=2 and row['F_Score'] <=2:
        return 'At Risk'
    else:
        return 'Other'
rfm['Segment']=rfm.apply(segment_customer,axis=1)
x=rfm[['R_Score','F_Score','M_Score']].astype(int)
y=rfm['Segment']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
clf=RandomForestClassifier(random_state=41)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
try:
    print("\nModel Performance on Test Set:")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy Score: {accuracy:.4f}")
except Exception as e:
    print(f"Error in model evaluation: {e}")
    exit()

#confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('Confusion Matrix for Customer Segment Prediction')
plt.xlabel('Predicted Segment')
plt.ylabel('True Segment')
plt.savefig('confusion_matrix.png')
plt.show()
plt.close()

#statistics
print("RFM Analysis Summary:")
print(rfm[['CustomerID','Recency','Frequency','Monetary','R_Score','F_Score','M_Score','Segment']])
print("\nSegment Distribution in Full Dataset:")
print(rfm['Segment'].value_counts())
#segment distributions
print("\nSegment Distribution in Training Set:")
print(y_train.value_counts())
print("\nSegment Distribution in Test Set;")
print(y_test.value_counts())
#visualization segment distribution
plt.figure(figsize=(10, 6))
segment_counts = rfm['Segment'].value_counts()
colors = ['red', 'lightpink', 'green', 'violet', 'orange']
ax = sns.countplot(
    data=rfm,
    x='Segment',
    hue='Segment',            
    order=segment_counts.index,
    palette=colors,
    legend=False               
)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')
plt.title('Customer Segment Distribution (Full Dataset)')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.savefig('segment_distribution_full.png')
plt.show()
plt.close()

#rfm score by histogram
fig, axes = plt.subplots(1, 3, figsize=(14, 6))

# Recency Histogram
sns.histplot(rfm['Recency'], bins=20, ax=axes[0], color='skyblue')
axes[0].set_title('Recency Distribution')
for p in axes[0].patches:
    if p.get_height() > 0:
        axes[0].annotate(int(p.get_height()), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=8)

# Frequency Histogram
sns.histplot(rfm['Frequency'], bins=20, ax=axes[1], color='red')
axes[1].set_title('Frequency Distribution')
for p in axes[1].patches:
    if p.get_height() > 0:
        axes[1].annotate(int(p.get_height()), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=8)

# Monetary Histogram
sns.histplot(rfm['Monetary'], bins=20, ax=axes[2], color='orange')
axes[2].set_title('Monetary Distribution')
for p in axes[2].patches:
    if p.get_height() > 0:
        axes[2].annotate(int(p.get_height()), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('rfm_distributions.png')
plt.show()
plt.close()

#scatter plot
palette = {
    'High-Value': 'skyblue',     
    'Loyal': 'lightpink',          
    'Big Spender': 'orange', 
    'At Risk': 'red',     
    'Other': 'green'           
}
# Custom color palette for segments
palette = {
    'High-Value': 'skyblue',    
    'Loyal': 'lightpink',       
    'Big Spender': 'orange',    
    'At Risk': 'red',        
    'Other': 'green'           
}

# Scatter plot with colored segments
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x='Recency', y='Monetary',
                hue='Segment', style='Segment', size='Frequency',
                palette=palette, sizes=(40, 200))
plt.title('Recency vs Monetary by Segment')
plt.xlabel('Recency (days)')
plt.ylabel('Monetary ($)')
plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('recency_monetary_scatter.png')
plt.show()
spending_summary = rfm.groupby('Segment').agg({
    'Monetary': ['sum', 'mean', 'count'],
}).reset_index()
spending_summary.columns = ['Segment', 'Total_Spending', 'Average_Spending', 'Customer_Count']

#Average amont spent by the customers
print("Spending Summary by Customer Segment:")
print(spending_summary[['Segment', 'Total_Spending', 'Average_Spending', 'Customer_Count']].round(2))
print("\nSegment with Highest Total Spending:", 
      spending_summary.loc[spending_summary['Total_Spending'].idxmax(), ['Segment', 'Total_Spending']].to_dict())
print("Segment with Highest Average Spending:", 
      spending_summary.loc[spending_summary['Average_Spending'].idxmax(), ['Segment', 'Average_Spending']].to_dict())
plt.figure(figsize=(10, 6))
segment_colors = ['skyblue', 'green', 'lightpink', 'red', 'purple']
ax = sns.barplot(data=spending_summary, x='Segment', y='Total_Spending', hue='Segment', palette=segment_colors, legend=False)
plt.title('Total Spending by Customer Segment')
plt.xlabel('Segment')
plt.ylabel('Total Spending ($)')
plt.xticks(rotation=45)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 0.5, f'{int(p.get_height())}', 
            ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()
