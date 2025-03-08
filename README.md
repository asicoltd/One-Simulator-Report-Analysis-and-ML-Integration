
# ONE Simulation, Data Processing, and Machine Learning Pipeline

This repository demonstrates a step-by-step pipeline from data creation to data processing and machine learning model evaluation for ONE simulation projects.

## Table of Contents

- [Overview](#overview)
- [1. Data Creation (Report Making)](#1-data-creation-report-making)
- [2. Data Processing](#2-data-processing)
- [3. Exploratory Data Analysis and Visualization](#3-exploratory-data-analysis-and-visualization)
- [4. Machine Learning](#4-machine-learning)
  - [4.1 Traditional Machine Learning](#41-traditional-machine-learning)
  - [4.2 Deep Learning with PyTorch](#42-deep-learning-with-pytorch)
- [5. Model Architecture Visualization](#5-model-architecture-visualization)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)

## Overview

This project covers the following steps:
1. **Data Creation:** Running simulations by dynamically updating simulation parameters and executing batch files to generate report files.
2. **Data Processing:** Parsing and processing report files, mapping data into a Pandas DataFrame, cleaning, and exporting the data.
3. **Exploratory Data Analysis:** Visualizing key relationships using pairplots and correlation heatmaps.
4. **Machine Learning:** Building and evaluating traditional (e.g., Random Forest) and deep learning models (using PyTorch) to predict target metrics.
5. **Model Architecture Visualization:** Visualizing the structure of the deep learning model using `torchviz`.

## 1. Data Creation (Report Making)

The simulation process involves:
- Running a compilation batch file (`compile.bat`) to set up the simulation environment.
- Dynamically updating simulation settings (e.g., end time, router type, number of hosts) in a settings file (`default_settings.txt`).
- Running the ONE simulator via a batch file (`one.bat`) in a loop over a specified duration.

```python
import datetime
import subprocess
import os

# Run compilation batch file
bat_file_path = r'compile.bat'
try:
    subprocess.run([bat_file_path], shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running the batch file: {e}")

# Simulation Parameters
Number_of_Hosts = 10
r = 1
totalHours = int(input('For how many hours?'))
totalLoops = 3 * 50 * totalHours  # 3 routers * 50 hosts (increment 10 each time) * total hours
new_value_end_time = 3600

for _ in range(totalLoops):
    file_path = 'default_settings.txt'
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Current time for naming
        current_datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Determine Router Type
        router_types = {1: 'EpidemicRouter', 2: 'SprayAndWaitRouter', 3: 'ProphetRouter'}
        formatted_router = router_types.get(r, 'EpidemicRouter')

        # Modify the settings file
        modified_lines = []
        for line in lines:
            if line.strip().startswith("Scenario.endTime ="):
                modified_lines.append(f"Scenario.endTime = {new_value_end_time}\n")
            elif line.strip().startswith("Scenario.name"):
                modified_lines.append(f"Scenario.name = {formatted_router}_{current_datetime}_hostNo{Number_of_Hosts}\n")
            elif line.strip().startswith("Group.router ="):
                modified_lines.append(f"Group.router = {formatted_router}\n")
            elif line.strip().startswith("Group.nrofHosts ="):
                modified_lines.append(f"Group.nrofHosts = {Number_of_Hosts}\n")
            elif line.strip().startswith("MovementModel.rngSeed ="):
                modified_lines.append(f"MovementModel.rngSeed = [1]\n")
            else:
                modified_lines.append(line)

        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        exit()

    # Run ONE Simulator
    one_bat_command = ["one.bat", "-b", "1"]
    try:
        subprocess.run(one_bat_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the batch file: {e}")

    # Update simulation parameters
    r += 1
    if r > 3:
        Number_of_Hosts += 10
        r = 1

    if Number_of_Hosts > 500:
        Number_of_Hosts = 10
        new_value_end_time += 3600  # Increase by 1 hour
```

## 2. Data Processing

This phase involves:
- Reading simulation report files from the specified folder.
- Parsing the files to extract relevant metrics.
- Mapping text keys to DataFrame column names.
- Export the processed data to a CSV file.

```python
import os
import pandas as pd

# Mapping of text keys to DataFrame column names
mapping = {
    'sim_time:': 'sim_time',
    'created:': 'created',
    'started:': 'started',
    'relayed:': 'relayed',
    'aborted:': 'aborted',
    'dropped:': 'dropped',
    'removed:': 'removed',
    'delivered:': 'delivered',
    'delivery_prob:': 'delivery_prob',
    'response_prob:': 'response_prob',
    'overhead_ratio:': 'overhead_ratio',
    'latency_avg:': 'latency_avg',
    'latency_med:': 'latency_med',
    'hopcount_avg:': 'hopcount_avg',
    'hopcount_med:': 'hopcount_med',
    'buffertime_avg:': 'buffertime_avg',
    'buffertime_med:': 'buffertime_med'
}

records = []
folder_path = "/reports"
for filename in os.listdir(folder_path):
    if filename.endswith(".txt") and os.path.getsize(os.path.join(folder_path, filename)) > 0:
        record = {}
        with open(os.path.join(folder_path, filename)) as file:
            for line in file:
                if 'Message' in line:
                    parts = line.split()
                    router_name = parts[4].split("_")[0]
                    record['router'] = 0 if router_name == 'EpidemicRouter' else (1 if router_name == 'ProphetRouter' else 2)
                    record['host'] = float(parts[4].split("_")[-1].replace("hostNo", ""))
                else:
                    for key, col in mapping.items():
                        if key in line:
                            record[col] = float(line.split()[1])
                            break
        records.append(record)

df = pd.DataFrame(records)
df.fillna(0, inplace=True)

# Export DataFrame to CSV
csv_path = "message_stats.csv"
df.to_csv(csv_path, index=False)
print(f"CSV file saved at: {csv_path}")
```

## 3. Exploratory Data Analysis and Visualization

After processing the data, the next steps include:
- Creating subsets for different router types.
- Generating pairplots and correlation heatmaps to explore relationships among features.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Mapping router numbers to names for visualization
router_mapping = {0: "EpidemicRouter", 1: "ProphetRouter", 2: "SprayAndWaitRouter"}
df["router"] = df["router"].replace(router_mapping)

# Pairplot for visualizing feature relationships
sns.pairplot(df, hue="router", vars=["host", "overhead_ratio", "latency_avg", "hopcount_avg", "buffertime_avg", "delivery_prob"])
plt.savefig("Pairplot.eps", format="eps", dpi=300, bbox_inches="tight")
plt.savefig("Pairplot.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()
```
![Plots](images/1%20Pairplot.png)
## 4. Machine Learning

### 4.1 Traditional Machine Learning

This approach uses models like RandomForestRegressor to predict target metrics.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prepare the dataset
X = df.drop(columns=['delivery_prob'])
y = df['delivery_prob']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred_rf)
print(f"RF Performance: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")
```
![Plots](images/6%20Scatter%20Plot%2C%20Residual%20Plot%2C%20Histogram%20of%20Residuals.png)
![Plots](images/10%20Corelation%20with%20Delivery_prob%20All%20Router.png)

### 4.2 Deep Learning with PyTorch

The deep learning approach involves building a neural network (IoVNet) to predict multiple target variables. Key steps include data normalization, model training, and evaluation.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define target and feature columns
target_cols = ['delivery_prob', 'latency_avg', 'overhead_ratio']
features_cols = list(set(df.columns) - set(target_cols + ['router']))

le = LabelEncoder()
df['router_type_enc'] = le.fit_transform(df['router'])
features_cols.append('router_type_enc')

X = df[features_cols].values.astype(np.float32)
y = df[target_cols].values.astype(np.float32)

scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_scaled, y_scaled = scaler_X.fit_transform(X), scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

class IoVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(IoVDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(IoVDataset(X_test, y_test), batch_size=64)

# Define the IoVNet model
class IoVNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IoVNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        return self.out(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IoVNet(X_train.shape[1], len(target_cols)).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-8)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=100)

num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

# Evaluate the model
model.eval()
y_true_list, y_pred_list = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        y_pred = model(batch_X).cpu().numpy()
        y_true_list.append(batch_y.cpu().numpy())
        y_pred_list.append(y_pred)

y_true, y_pred = np.vstack(y_true_list), np.vstack(y_pred_list)
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
print(f"Performance Metrics:\nMSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")
```
![Plots](images/11%20NN%20model%20architecture.png)

## 5. Model Architecture Visualization

Visualize the neural network architecture using `torchviz`:

```python
from torchviz import make_dot
dummy_input = torch.randn(1, X_train.shape[1]).to(device)
dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
dot.render("iov_model_architecture", format="jpg")  # Save as JPG
dot.render("iov_model_architecture", format="eps")  # Save as EPS
dot  # Display the network graph
```
![Plots](images/12%20learning%20curve.png)
![Plots](images/13%20performance.png)
## Requirements

Install the necessary Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch torchvision torchviz statsmodels
```

## Execution

Download ReportAnalysis2.ipynb at One Simulator Reports Folder And Run it
