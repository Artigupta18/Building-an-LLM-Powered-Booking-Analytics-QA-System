#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# File: evaluate_qa.py
import requests
import pandas as pd
from typing import Dict, Union, Callable 

# API endpoint
BASE_URL = "http://127.0.0.1:8000/ask"

# Load data for ground truth
print("Loading data...")
try:
    data = pd.read_csv('hotel_bookings_with_embeddings.csv')
except Exception as e:
    print(f"Error loading CSV: {e}")
    raise
print("Converting dates...")
try:
    data['arrival_date'] = pd.to_datetime(data['arrival_date'])
except Exception as e:
    print(f"Error converting dates: {e}")
    raise

# Test queries and expected answers
TEST_QUERIES = [
    {
        "query": "What’s the total revenue for July 2016?",
        "expected": lambda d: {"Total Revenue": f"${d[(d['arrival_date_month'] == 'July') & (d['arrival_date_year'] == 2016) & (d['is_canceled'] == 0)]['revenue'].sum():.2f}", "Month": "July", "Year": "2016"}
    },
    {
        "query": "What’s the cancellation rate?",
        "expected": lambda d: {"Cancellation Rate": f"{d['is_canceled'].mean() * 100:.2f}%"}
    },
    {
        "query": "Which country has the highest booking cancellations?",
        "expected": lambda d: {"Location with Highest Cancellations": d[d['is_canceled'] == 1]['country'].value_counts().idxmax(), "Total Cancellations": int(d[d['is_canceled'] == 1]['country'].value_counts().max())}
    },
    {
        "query": "Why do people cancel bookings?",
        "expected": "qualitative"
    },
    {
        "query": "What’s the busiest month for bookings?",
        "expected": "qualitative"
    }
]

def evaluate_query(query: str, expected: Union[Callable, str]) -> Dict:
    print(f"Evaluating query: {query}")
    try:
        response = requests.post(BASE_URL, json={"question": query})
        print(f"API Status Code: {response.status_code}")
        if response.status_code != 200:
            return {"query": query, "status": "error", "response": response.text}
        actual = response.json()
        print(f"API Response: {actual}")
    except Exception as e:
        return {"query": query, "status": "error", "response": str(e)}

    if callable(expected):  # Quantitative check
        expected_output = expected(data)
        is_correct = actual == expected_output
        return {"query": query, "status": "pass" if is_correct else "fail", "expected": expected_output, "actual": actual}
    else:  # Qualitative check
        return {"query": query, "status": "review", "response": actual}

def run_evaluation():
    results = []
    for test in TEST_QUERIES:
        result = evaluate_query(test["query"], test["expected"])
        results.append(result)
        print(f"Query: {result['query']}")
        if result["status"] == "review":
            print(f"Response (review manually): {result['response']}")
        else:
            print(f"Status: {result['status']}")
            if result["status"] == "fail":
                print(f"Expected: {result['expected']}")
                print(f"Actual: {result['actual']}")
        print("-" * 50)
    return results

if __name__ == "__main__":
    print("Running Q&A Accuracy Evaluation...")
    run_evaluation()

