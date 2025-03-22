#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# File: benchmark_api.py
import requests
import time
from statistics import mean, stdev

# API endpoints
ANALYTICS_URL = "http://127.0.0.1:8000/analytics"
ASK_URL = "http://127.0.0.1:8000/ask"

# Test cases
ANALYTICS_TESTS = [
    {"report_type": "revenue_trends"},
    {"report_type": "cancellation_rate"},
    {"report_type": "top_locations"},
    {"report_type": "lead_time_distribution"}
]

ASK_TESTS = [
    {"question": "What’s the total revenue for July 2016?"},
    {"question": "What’s the cancellation rate?"},
    {"question": "Why do people cancel bookings?"},
    {"question": "What’s the busiest month for bookings?"}
]

def measure_time(url: str, payload: dict, iterations: int = 5) -> tuple:
    times = []
    for _ in range(iterations):
        start = time.time()
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            times.append(time.time() - start)
        else:
            print(f"Error for {payload}: {response.text}")
            return None, None
    return mean(times), stdev(times)

def benchmark_api():
    print("Benchmarking /analytics Endpoint...")
    for test in ANALYTICS_TESTS:
        avg_time, std_dev = measure_time(ANALYTICS_URL, test)
        if avg_time:
            print(f"Report: {test['report_type']}, Avg Time: {avg_time:.3f}s, Std Dev: {std_dev:.3f}s")
    print("-" * 50)

    print("Benchmarking /ask Endpoint...")
    for test in ASK_TESTS:
        avg_time, std_dev = measure_time(ASK_URL, test)
        if avg_time:
            print(f"Question: {test['question']}, Avg Time: {avg_time:.3f}s, Std Dev: {std_dev:.3f}s")
    print("-" * 50)

if __name__ == "__main__":
    benchmark_api()

