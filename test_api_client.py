import requests
import time

API_URL = "http://127.0.0.1:8000/predict"

def generate_test_payloads():
    """Generates a list of test payloads matching the new string-based input format."""
    return [
        {
            "Present_Price": 5.59,
            "Kms_Driven": 27000,
            "Fuel_Type": "Petrol",
            "Seller_Type": "Dealer",
            "Transmission": "Manual",
            "Owner": 0,
            "age": 7
        },
        {
            "Present_Price": 12.50,
            "Kms_Driven": 45000,
            "Fuel_Type": "Diesel",
            "Seller_Type": "Individual",
            "Transmission": "Manual",
            "Owner": 1,
            "age": 4
        },
        {
            "Present_Price": 30.00,
            "Kms_Driven": 12000,
            "Fuel_Type": "Petrol",
            "Seller_Type": "Dealer",
            "Transmission": "Automatic",
            "Owner": 0,
            "age": 2
        }
    ]

def stress_test_api(num_requests=10):
    payloads = generate_test_payloads()
    print(f"Sending {num_requests} requests to {API_URL}...\n")
    
    success_count = 0
    start_time = time.time()
    
    for i in range(num_requests):
        # Rotate through the payloads
        payload = payloads[i % len(payloads)]
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                success_count += 1
                data = response.json()
                print(f"Request {i+1} Succeded | Price: {data.get('predicted_price_lakhs')} {data.get('currency')}")
            else:
                print(f"Request {i+1} Failed: {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"Request {i+1} Failed: Connection Refused. Is the API running?")
            break
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- API Test Results ---")
    print(f"Successful Requests: {success_count}/{num_requests}")
    print(f"Total Time: {duration:.4f} seconds")
    print(f"Average Request Time: {(duration/num_requests):.4f} seconds")

if __name__ == "__main__":
    stress_test_api(10)
