import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import csv
import random
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import rioxarray
import os
import numpy as np
import pickle
import matplotlib.patches as mpatches
from matplotlib import colors
import glob
import leafmap.foliumap as leafmap
import rasterio

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 150}

plt.rc('font', **font)

st.title('Random Forest Model Interactive Visualization')
st.write('This app allows you to adjust parameters of a Random Forest model and observe its performance.')

# Handle "No Limit" selection for n_estimators
n_estimators = st.sidebar.selectbox('Number of Trees', [100, 200, 300, 'No Limit'])
if n_estimators == 'No Limit':
    n_estimators = 1000  # Set to a large number or adjust as needed
max_depth = st.sidebar.slider('Max Depth', 1, 100, 10)


@st.cache_data
def efficient_reservoir_sample(csv_file, sample_size, chunk_size=10000):
    reservoir = []
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        # Filter out rows with all zeros
        filtered_chunk = chunk[~(chunk == 0).all(axis=1)]
        
        for _, row in filtered_chunk.iterrows():
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = random.randint(0, len(reservoir)-1)
                if j < sample_size:
                    reservoir[j] = row

    sampled_df = pd.DataFrame(reservoir)
    return sampled_df

# Adjust the file path and sample size as needed
file_path = '~/Downloads/streamlit.csv'
sample_size = int(128281768 * 0.00001)
sampled_df = efficient_reservoir_sample(file_path, sample_size)

# Splitting the data
X = sampled_df.drop('y', axis=1)
y = sampled_df['y']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model

min_samples_split = st.sidebar.slider('Minimum Samples Split', 2, 10, 2)
min_samples_leaf = st.sidebar.slider('Minimum Samples Leaf', 1, 10, 1)
bootstrap = st.sidebar.checkbox('Bootstrap Samples', value=True)
criterion = st.sidebar.selectbox('Criterion', ['gini', 'entropy'])
max_features = st.sidebar.selectbox('Max Features', [None, 'sqrt', 'log2'])


rf_clf = RandomForestClassifier(
    n_estimators=n_estimators, 
    max_depth=max_depth, 
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    bootstrap=bootstrap,
    criterion=criterion,
    max_features=max_features,
    random_state=42
)

rf_clf.fit(x_train, y_train)

# Make predictions
y_pred = rf_clf.predict(x_test)

# Calculate and display metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
iou = jaccard_score(y_test, y_pred)

# Visualize the metrics in a bar chart
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'IoU'],
    'Value': [accuracy, precision, recall, f1, iou]
})


# Use Streamlit's columns to display the table and the bar chart side by side
col1, col2 = st.columns(2)

with col1:
    st.write("Summary Table of Metrics")
    st.write(metrics_df.set_index('Metric'))

with col2:
    st.write("Bar Chart of Metrics")
    fig, ax = plt.subplots()
    ax.bar(metrics_df['Metric'], metrics_df['Value'])
    ax.set_ylim(0, 1)  # Adjust as needed
    st.bar_chart(metrics_df.set_index('Metric'))

feature_importances = pd.Series(rf_clf.feature_importances_, index=X.columns)

# Creating a DataFrame for the bar chart
feature_importance_df = pd.DataFrame({
    'Feature': feature_importances.index, 
    'Importance': feature_importances.values
}).set_index('Feature')

st.write('Bar Chart of Feature Importance')
st.bar_chart(feature_importance_df)


#--------------------------------image plotting stuff

#load model
model = pickle.load(open('/Users/spotter/Downloads/rf_model_dnbr_2.pickle', 'rb'))


in_path = '/Users/spotter/Downloads'

# List of files
file_list = glob.glob(in_path + '/*.tif')

file_list = [i for i in file_list if 'median' in i]


# Allow the user to select a file from the list
selected_file = st.selectbox("Select a Fire to Predict On:", file_list)

#get center for mapping
 # Get the bounding box of the raster
with rasterio.open(selected_file) as src:
    bounds = src.bounds

# Calculate the center of the bounding box
center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]

in_file = rioxarray.open_rasterio(os.path.join(in_path, selected_file)).to_numpy()

in_file_ras = rioxarray.open_rasterio(os.path.join(in_path, selected_file)).isel(band = 9)

in_file_ras_pred = rioxarray.open_rasterio(os.path.join(in_path, selected_file)).isel(band = 2)

# Extract band names from metadata
# band_names = in_file_ras.band.attrs["long_name"]

reshaped_data = in_file.reshape(in_file.shape[0], -1)

band_names = ['blue', 'green', 'red', 'NIR', 'SWIR1', 'SWIR2', 'dNBR', 'dNDVI', 'dNDII', 'y']


# Create a DataFrame
training = pd.DataFrame(reshaped_data.T, columns=band_names)

#original values were originally scaled
columns_to_divide = [col for col in training.columns if col != 'y']

# # Divide selected columns by 1000
training[columns_to_divide] = training[columns_to_divide].div(1000).round(3)

# #just dnbr
for_predict = training[['dNBR']]

# #predict
predict = model.predict(for_predict)

# #turn back to image
predict_image = predict.reshape(in_file.shape[1], in_file.shape[2])

#add the predicted image to geospatial image
in_file_ras_pred.data = predict_image

in_file_ras_pred.attrs['long_name'] = ['predicted']

in_file_ras.attrs['long_name'] = ['y']


out_path_ext = selected_file.split('/')[-1].replace('median_', 'predicted_')
out_path_ext_y = selected_file.split('/')[-1].replace('median_', 'y_')


full_out = os.path.join(in_path, out_path_ext)
full_out_y = os.path.join(in_path, out_path_ext_y)

in_file_ras_pred.rio.to_raster(full_out)
in_file_ras.rio.to_raster(full_out_y)


#display images side by side
# col3, col4 = st.columns(2)
col3 = st.columns(1)



plt.subplots_adjust(wspace=0.1, hspace=0)
col3[0].write("Comparison of Reference Data and Predicted Fire Locations")

m = leafmap.Map(lcenter=center, zoom=12)
m.add_raster(selected_file, bands=[6], layer_name = 'dNBR', colormap='viridis', vmin = -1, vmax = 1)
m.add_raster(full_out_y, bands=[1], layer_name = 'Reference Fire', colormap={0: 'gray', 1: 'red'})
m.add_raster(full_out, bands = [1], layer_name = 'Predicted Fire', colormap={0: 'gray', 1: 'red'})
m.to_streamlit()

# with col3:
#     st.write("Comparison of Reference Data and Predicted Fire Locations")
#     # st.pyplot(fig)

#     m = leafmap.Map(lcenter=center, zoom=12)
#     m.add_raster(selected_file, bands=[9], layer_name = 'Reference Fire', colormap='viridis')
#     m.add_raster(full_out, bands = [1], layer_name = 'Predicted Fire', colormap='viridis')
#     m.to_streamlit()





# for f in file_list:

#     in_file_ras = rioxarray.open_rasterio(f).isel(band = 9)


#     in_file_ras.attrs['long_name'] = ['y']


#     out_path_ext_y = f.split('/')[-1].replace('median_', 'y_')


#     full_out_y = os.path.join(in_path, out_path_ext_y)

#     in_file_ras.rio.to_raster(full_out_y)