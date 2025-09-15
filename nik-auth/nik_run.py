import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# Initialize Firebase Admin SDK with your service account credentials
cred = credentials.Certificate("nik-auth/firebase.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Read NIK data from Excel
df_nik = pd.read_excel("nik-auth/nik_desa_rahayu.xlsx")

# Reference to the Firestore collection
nik_collection = db.collection('nik-auth')

# Insert data into Firestore
for index, row in df_nik.iterrows():
    nik = row['NIK']
    # Add NIK to Firestore
    nik_collection.add({'NIK': nik})

print("Data uploaded successfully!")
