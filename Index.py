import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import UnivariateSpline

# For GAM
from pygam import LinearGAM, s

# For LIME
from lime import lime_tabular

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Columns in the dataset:", df.columns.tolist()) 
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price'] 
    df = df[df['Quantity'] > 0]  # Remove negative values
    
    # Group by Customer ID for RFM analysis
    customer_df = df.groupby('Customer ID').agg({
        'InvoiceDate': [lambda x: (datetime(2011, 12, 10) - x.max()).days,  # Recency
                        'count'],                                          # Frequency
        'TotalPrice': 'sum'                                                 # Monetary value
    }).reset_index()

    customer_df.columns = ['CustomerID', 'Recency', 'Frequency', 'MonetaryValue']
    customer_df['MonetaryValue'] = customer_df['MonetaryValue'].clip(upper=customer_df['MonetaryValue'].quantile(0.95))
    
    return customer_df

# Train the GAM model for predicting monetary value and CLV
def train_gam(customer_df):
    # GAM for predicting monetary value
    gam_mv = LinearGAM(s(0) + s(1)).fit(customer_df[['Recency', 'Frequency']], customer_df['MonetaryValue'])
    customer_df['PredictedMonetaryValue'] = gam_mv.predict(customer_df[['Recency', 'Frequency']])
    
    # Predict churn probability
    gam_churn = LinearGAM(s(0) + s(1)).fit(customer_df[['Recency', 'Frequency']], np.random.uniform(0.5, 1.5, len(customer_df)))
    customer_df['ChurnProbability'] = gam_churn.predict(customer_df[['Recency', 'Frequency']])
    
    # Predict CLV 
    customer_df['CLV'] = customer_df['PredictedMonetaryValue'] * customer_df['ChurnProbability']
    
    return customer_df, gam_mv

# Plot the relationship between Frequency/Recency vs. Monetary Value and CLV
def plot_binned_relationship(customer_df):
    # Binning Recency and Frequency into intervals
    customer_df['Recency_Binned'] = pd.cut(customer_df['Recency'], bins=np.arange(0, customer_df['Recency'].max(), 25))
    customer_df['Frequency_Binned'] = pd.cut(customer_df['Frequency'], bins=np.arange(0, customer_df['Frequency'].max(), 100)) 

    # Aggregate data by binned Recency and Frequency
    recency_agg = customer_df.groupby('Recency_Binned').agg({'MonetaryValue': 'mean', 'CLV': 'mean'}).reset_index()
    frequency_agg = customer_df.groupby('Frequency_Binned').agg({'MonetaryValue': 'mean', 'CLV': 'mean'}).reset_index()

    # Plot the binned Recency vs. Monetary Value and CLV
    plt.figure(figsize=(16, 8))  

    # Plot Recency vs Monetary Value
    plt.subplot(1, 2, 1)
    plt.plot(recency_agg['Recency_Binned'].astype(str), recency_agg['MonetaryValue'], marker='o', color='blue')
    plt.title('Effect of Recency on Average Monetary Value')
    plt.xlabel('Recency (Binned)')
    plt.ylabel('Average Monetary Value')
    plt.xticks(rotation=90)

    # Plot Frequency vs Monetary Value
    plt.subplot(1, 2, 2)
    plt.plot(frequency_agg['Frequency_Binned'].astype(str), frequency_agg['MonetaryValue'], marker='o', color='red')
    plt.title('Effect of Frequency on Average Monetary Value')
    plt.xlabel('Frequency (Binned)')
    plt.ylabel('Average Monetary Value')
    plt.xticks(rotation=90)  
    plt.xticks(np.arange(0, len(frequency_agg['Frequency_Binned']), 5), frequency_agg['Frequency_Binned'][::5])  # Show every 5th label

    plt.tight_layout()
    plt.show()

    # Plot the binned Recency vs CLV
    plt.figure(figsize=(16, 8))  # Increase figure width even more

    # Plot Recency vs CLV
    plt.subplot(1, 2, 1)
    plt.plot(recency_agg['Recency_Binned'].astype(str), recency_agg['CLV'], marker='o', color='blue')
    plt.title('Effect of Recency on Average CLV')
    plt.xlabel('Recency (Binned)')
    plt.ylabel('Average CLV')
    plt.xticks(rotation=90)

    # Plot Frequency vs CLV
    plt.subplot(1, 2, 2)
    plt.plot(frequency_agg['Frequency_Binned'].astype(str), frequency_agg['CLV'], marker='o', color='red')
    plt.title('Effect of Frequency on Average CLV')
    plt.xlabel('Frequency (Binned)')
    plt.ylabel('Average CLV')
    plt.xticks(rotation=90)
    plt.xticks(np.arange(0, len(frequency_agg['Frequency_Binned']), 5), frequency_agg['Frequency_Binned'][::5])  # Show every 5th label

    plt.tight_layout()
    plt.show()

# Apply LIME for explainability on Monetary Value and CLV
def apply_lime(customer_df, gam_mv, gam_clv):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=customer_df[['Recency', 'Frequency']].values,
        feature_names=['Recency', 'Frequency'],
        class_names=['Predicted Monetary Value', 'Predicted CLV'],
        mode='regression'
    )

    # Choose an instance to explain (e.g., the first instance)
    instance = customer_df[['Recency', 'Frequency']].iloc[0].values

    # Explanation for monetary value
    print("\nLIME Explanation for Predicted Monetary Value:")
    explanation_mv = explainer.explain_instance(instance, gam_mv.predict, num_features=2)
    for feature, weight in explanation_mv.as_list():
        print(f"{feature}: {weight}")

    # Explanation for CLV
    print("\nLIME Explanation for Predicted CLV:")
    explanation_clv = explainer.explain_instance(instance, gam_clv.predict, num_features=2)
    for feature, weight in explanation_clv.as_list():
        print(f"{feature}: {weight}")

    return customer_df

# Main function to execute the steps
def main(file_path):
    customer_df = load_data(file_path)
    customer_df, gam_mv = train_gam(customer_df)

    # Output the predictions
    print("\nCLV Predictions:")
    print(customer_df[['CustomerID', 'Recency', 'Frequency', 'MonetaryValue', 'PredictedMonetaryValue', 'ChurnProbability', 'CLV']].head(10))

    # Plot binned Frequency and Recency against Monetary Value and CLV
    plot_binned_relationship(customer_df)

    # Create another GAM for CLV for LIME explanations
    gam_clv = LinearGAM(s(0) + s(1)).fit(customer_df[['Recency', 'Frequency']], customer_df['CLV'])

    # Apply LIME explainability
    customer_df = apply_lime(customer_df, gam_mv, gam_clv)

    # Optionally save the customer data
    customer_df.to_csv('customer_clv_monetary_value_predictions.csv', index=False)


file_path = 'C:/Users/Asus/OneDrive/Desktop/CLV/online_retail_II.csv'
main(file_path)
