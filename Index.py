# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
from pygam import LinearGAM, s
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lifelines import CoxPHFitter
from lime import lime_tabular


# Load the dataset
file_path = 'C:/Users/Asus/OneDrive/Desktop/CLV/online_retail_II.csv'
df = pd.read_csv(file_path)

# Preprocessing the data
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['Price']

# Remove negative values (likely due to canceled orders)
df = df[df['Quantity'] > 0]

# Aggregate data at the customer level for RFM and monetary value calculation
customer_df = df.groupby('Customer ID').agg({
    'InvoiceDate': [lambda x: (datetime(2011, 12, 10) - x.max()).days,  # Recency
                    'count'],                                          # Frequency
    'TotalPrice': 'sum'                                                 # Monetary value
}).reset_index()

# Rename columns
customer_df.columns = ['CustomerID', 'Recency', 'Frequency', 'MonetaryValue']

# Remove outliers by capping the extreme values (optional)
customer_df['MonetaryValue'] = customer_df['MonetaryValue'].clip(upper=customer_df['MonetaryValue'].quantile(0.95))


### Function for GAM + Cox Proportional Hazards Model

def fit_gam_and_coxph(customer_df):
    # 1. Predict Monetary Value using GAM
    gam_mv = LinearGAM(s(0) + s(1))  # Define GAM model with two features only
    X_mv = customer_df[['Recency', 'Frequency']]  # Independent variables
    y_mv = customer_df['MonetaryValue']  # Target variable

    # Fit the GAM model
    gam_mv.fit(X_mv, y_mv)

    # Predict Monetary Value
    customer_df['PredictedMonetaryValue'] = gam_mv.predict(X_mv)

    # Predict Recency and Frequency using GAM
    gam_recency = LinearGAM(s(0) + s(1)).fit(X_mv, customer_df['Recency'])
    customer_df['PredictedRecency'] = gam_recency.predict(X_mv)

    gam_frequency = LinearGAM(s(0) + s(1)).fit(X_mv, customer_df['Frequency'])
    customer_df['PredictedFrequency'] = gam_frequency.predict(X_mv)

    # 2. Predict Churn using Cox Proportional Hazards Model
    # Prepare data for survival analysis (churn prediction)
    df_churn = customer_df.copy()
    df_churn['Churned'] = np.where(customer_df['Recency'] > customer_df['Recency'].quantile(0.75), 1, 0)  # Example logic
    df_churn['Duration'] = customer_df['Recency']  # Use Recency as the time duration for churn prediction

    # Fit the Cox Proportional Hazards Model
    cph = CoxPHFitter()
    cph.fit(df_churn[['Duration', 'Frequency', 'MonetaryValue', 'Churned']], duration_col='Duration', event_col='Churned')

    # Predict Churn probability
    customer_df['ChurnProbability'] = cph.predict_partial_hazard(df_churn[['Duration', 'Frequency', 'MonetaryValue']])

    # 3. Compute CLV
    customer_df['CLV'] = (customer_df['PredictedMonetaryValue'] * customer_df['PredictedFrequency']) / (1 + customer_df['ChurnProbability'])

    return customer_df, gam_mv  # Return the trained GAM model


### PyTorch NAM Model Definition

class NAM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NAM, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.hidden(x)


### Function for PyTorch NAM implementation

def fit_nam(customer_df):
    # Define NAM model
    input_size = 2  # Recency and Frequency
    hidden_size = 64
    output_size = 1
    model = NAM(input_size, hidden_size, output_size)
    
    # Data preparation
    X = customer_df[['Recency', 'Frequency']].values
    y = customer_df['MonetaryValue'].values
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # DataLoader for PyTorch
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    epochs = 100
    for epoch in range(epochs):
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Predictions
    with torch.no_grad():
        predictions = model(X_tensor).numpy().flatten()
        customer_df['PredictedMonetaryValue_NAM'] = predictions
    
    return customer_df, model  # Return the trained NAM model


### Function to Compare Interpretability with LIME
def compare_with_lime(customer_df, gam_model, nam_model):
    # Independent variables
    X_mv = customer_df[['Recency', 'Frequency']].values
    
    # Define the LIME explainer for GAM predictions
    explainer_gam = lime_tabular.LimeTabularExplainer(X_mv, mode='regression')

    # Define the LIME explainer for NAM predictions
    explainer_nam = lime_tabular.LimeTabularExplainer(X_mv, mode='regression')

    # Select an instance for comparison (e.g., the first row)
    instance = X_mv[0]

    # Explain the prediction from the GAM model
    exp_gam = explainer_gam.explain_instance(instance, gam_model.predict)
    print("LIME Explanation for GAM Model:")
    print(exp_gam.as_list())  # Print explanation for GAM

    # Define a wrapper to convert the NumPy array to a Tensor before passing it to NAM
    def nam_predict(input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)  # Convert to PyTorch tensor
        return nam_model(input_tensor).detach().numpy()

    # Explain the prediction from the NAM model
    exp_nam = explainer_nam.explain_instance(instance, nam_predict)
    print("\nLIME Explanation for NAM Model:")
    print(exp_nam.as_list())  # Print explanation for NAM

    # Optionally return the explanations
    return exp_gam, exp_nam




### Main Execution

# 1. Run GAM and CoxPH Model
customer_df, gam_mv = fit_gam_and_coxph(customer_df)

# 2. Run PyTorch NAM Model
customer_df, nam_model = fit_nam(customer_df)

# 3. Compare Interpretability using LIME (pass the trained models)
X_mv = customer_df[['Recency', 'Frequency']].values
exp_gam, exp_nam = compare_with_lime(customer_df, gam_mv, nam_model)  # Passing PyTorch NAM as `model`

# Optionally save the result
customer_df.to_csv('C:/Users/Asus/OneDrive/Desktop/CLV/customer_clv_predictions.csv', index=False)

# Display the final dataframe with CLV predictions
print(customer_df[['CustomerID', 'PredictedMonetaryValue', 'PredictedMonetaryValue_NAM', 'CLV']])
