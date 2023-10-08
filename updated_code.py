import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import geopandas as gpd
import matplotlib.pyplot as plt
import math

# Function to fetch NASA's fire data
def fetch_fire_data():
    url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/c4f9c87a128b458b6a69d96cfc2e3e60/VIIRS_SNPP_NRT/world/1/2023-10-07"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = pd.read_csv(url)
        return data
    else:
        print("Failed to fetch data.")
        return None

# Function to preprocess data
def preprocess_data(data):
    # relevant_columns = ['latitude', 'longitude', 'confidence', 'bright_ti4', 'acq_date']
    relevant_columns = ['latitude', 'longitude', 'bright_ti4','frp', 'acq_date']
    data = data[relevant_columns]
    data['acq_date'] = pd.to_datetime(data['acq_date'])
    data['is_fire'] = 1  # Simulated: All data points are treated as fire incidents
    return data

# Function to train a machine learning model
def train_model(data):
    # X = data[['latitude', 'longitude', 'confidence', 'bright_ti4']]
    X = data[['latitude', 'longitude', 'bright_ti4']]
    y = data[['frp']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return clf, accuracy

# Function to send email alerts
def send_email_alert(sender_email, sender_password, receiver_email):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'Fire Detection Alert'
    
    body = 'Fire detected in the area. Please take appropriate action.'
    msg.attach(MIMEText(body, 'plain'))
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()
def calculate_damaged_area_coordinates(latitude, longitude, fire_radius_km):
    # Earth's radius in kilometers (mean radius)
    earth_radius_km = 6371.0

    # Convert fire radius from kilometers to degrees (approximation)
    fire_radius_deg = fire_radius_km / earth_radius_km * (180.0 / math.pi)

    # Generate coordinates for the circular damaged area
    num_points = 360  # Number of points to generate for the circle
    step = 360.0 / num_points  # Angle step between points

    # Initialize lists to store latitude and longitude coordinates
    damaged_area_latitudes = []
    damaged_area_longitudes = []

    # Generate coordinates for the circular area
    for angle in range(0, 360, int(step)):
        angle_rad = math.radians(angle)
        lat = latitude + (fire_radius_deg * math.sin(angle_rad))
        lon = longitude + (fire_radius_deg * math.cos(angle_rad))
        damaged_area_latitudes.append(lat)
        damaged_area_longitudes.append(lon)

    return damaged_area_latitudes, damaged_area_longitudes

# Function to visualize fire data on a map
def visualize_fire_data_on_map(data):
    gdf = gpd.GeoDataFrame(data, 
                           geometry=gpd.points_from_xy(data.longitude, data.latitude))
    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    world.boundary.plot(ax=ax, linewidth=1, color='gray')
    gdf.plot(ax=ax, markersize=5, color='red', label='Fire Incidents')
    
    plt.title('Fire Incidents on Map')
    plt.legend()
    plt.show()

# Main function
if __name__ == '__main__':
    # Simulate fetching NASA's fire data
    fire_data = fetch_fire_data()
    
    if fire_data is not None:
        # Preprocess data
        processed_data = preprocess_data(fire_data)
        
        # Train a machine learning model
        # model, accuracy = train_model(processed_data)
        #model, accuracy = train_model(fire_data)
        
        print(f"Fire Confidence: {processed_data['frp']}")
        
        # Check if there's a significant fire incident (dummy threshold for demonstration)
        # if accuracy > 0.8:
        #     sender_email = 'your_email@gmail.com'
        #     sender_password = 'your_email_password'
        #     receiver_email = 'receiver_email@gmail.com'
            
        #     # send_email_alert(sender_email, sender_password, receiver_email)
            
        # Visualize fire data on a map
        visualize_fire_data_on_map(processed_data)
