import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Fetch NASA's fire data (or use a local CSV file)
url = "url"
response = requests.get(url)

if response.status_code == 200:
    data = pd.read_csv(response.text)
else:
    print("Failed to fetch data.")

# Data Preprocessing
relevant_columns = ['latitude', 'longitude', 'confidence', 'brightness', 'acq_date']
data = data[relevant_columns]
data['acq_date'] = pd.to_datetime(data['acq_date'])
data['is_fire'] = np.random.choice([0, 1], size=len(data))

# Split data for training and testing
X = data[['latitude', 'longitude', 'confidence', 'brightness']]
y = data['is_fire']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Alerting System
if accuracy > 0.8:
    sender_email = 'your_email@gmail.com'
    sender_password = 'your_email_password'
    receiver_email = 'receiver_email@gmail.com'

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
