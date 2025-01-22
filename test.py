import requests

# URL of the API
url = "http://127.0.0.1:5000/predict"

# Original data with 41 features (modify slightly for testing)
features = [
    500.8659,  # Amount (modified)
    2023,  # Extracted year from Date
    11,  # Extracted month from Date
    9,  # Extracted day from Date
    0,  # Transaction Type: "Debit" (after encoding)
    1,  # MCC Category: "Electronics" (after encoding)
    0,  # Location: "US" (after encoding)
    1,  # Device: "Desktop" (after encoding)
    4000.6439,  # Balance Before Transaction (modified)
    13,  # Customer Age
    0.105,  # Velocity (modified)
    71,  # Credit Score
    125000.0,  # Customer Income (modified)
    14000.0,  # Spending Patterns (modified)
    542,  # Transactions History
    1,  # Merchant Reputation: "Good" (after ordinal encoding)
    7,  # Online Transaction Frequency
    1300.0,  # Card Limit (modified)
    2,  # Spending Category: "Medium" (after ordinal encoding)
    # Remaining features after one-hot encoding and scaling (examples)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Placeholder for one-hot encoded values
    0.1, 0.5, 0.8, 1.2, 2.5,  # Placeholder for scaled features
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

# Prepare JSON payload
data = {"features": features}

# Send POST request to the API
response = requests.post(url, json=data)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
