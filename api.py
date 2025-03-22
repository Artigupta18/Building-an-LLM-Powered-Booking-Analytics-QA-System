#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# File: api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import re
import os

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "your_gemini_key"))  # Replace with your key
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Load data and FAISS index
data = pd.read_csv('hotel_bookings_with_embeddings.csv')
data['arrival_date'] = pd.to_datetime(data['arrival_date'])
index = faiss.read_index('hotel_booking_index.faiss')

# Initialize FastAPI app
app = FastAPI(title="Hotel Booking Analytics API")

# Pydantic models for request bodies
class AnalyticsRequest(BaseModel):
    report_type: str  # e.g., "revenue_trends", "cancellation_rate"

class AskRequest(BaseModel):
    question: str

# Lazy-load SentenceTransformer model
def get_model():
    if not hasattr(get_model, 'model'):
        get_model.model = SentenceTransformer('all-MiniLM-L6-v2')
    return get_model.model

# Analytics logic
def generate_analytics(report_type: str):
    report_type = report_type.lower()
    if report_type == "revenue_trends":
        revenue_trends = data[data['is_canceled'] == 0].groupby(
            data['arrival_date'].dt.to_period('M')
        )['revenue'].sum().reset_index()
        revenue_trends['arrival_date'] = revenue_trends['arrival_date'].dt.to_timestamp()
        return {"revenue_trends": revenue_trends.to_dict(orient='records')}

    elif report_type == "cancellation_rate":
        cancellation_rate = data['is_canceled'].mean() * 100
        return {"cancellation_rate": f"{cancellation_rate:.2f}%"}

    elif report_type == "top_locations":
        geo_distribution = data[data['is_canceled'] == 0]['country'].value_counts().head(10)
        return {"top_locations": geo_distribution.to_dict()}

    elif report_type == "lead_time_distribution":
        lead_time_bins = pd.cut(data['lead_time'], bins=10).value_counts().sort_index()
        return {"lead_time_distribution": lead_time_bins.to_dict()}

    else:
        raise HTTPException(status_code=400, detail="Invalid report type. Options: revenue_trends, cancellation_rate, top_locations, lead_time_distribution")

# Q&A logic
def ask_gemini(query, context_data):
    prompt = f"""
    You are a hotel booking analytics assistant. Use the provided data to answer the query.
    Insights available: revenue trends, cancellation rates, lead time, booking locations.

    User Query: {query}
    Relevant Data: {context_data}

    Provide a clear and concise response.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and response.text else "No relevant information found."
    except Exception as e:
        return f"Error: {str(e)}"

def handle_analytics_query(query):
    query_lower = query.lower()
    month_map = {
        'january': 'January', 'february': 'February', 'march': 'March', 'april': 'April',
        'may': 'May', 'june': 'June', 'july': 'July', 'august': 'August', 'september': 'September',
        'october': 'October', 'november': 'November', 'december': 'December'
    }

    if "total revenue" in query_lower:
        match = re.search(r'(\w+)\s(\d{4})', query)
        if match:
            month, year = match.groups()
            month_name = month_map.get(month.lower())
            filtered_data = data[(data['arrival_date_month'] == month_name) & 
                                (data['arrival_date_year'] == int(year)) & 
                                (data['is_canceled'] == 0)]
            total_revenue = filtered_data['revenue'].sum()
            return {"Total Revenue": f"${total_revenue:.2f}", "Month": month_name, "Year": year}

    elif "cancellation rate" in query_lower:
        cancellation_rate = data['is_canceled'].mean() * 100
        return {"Cancellation Rate": f"{cancellation_rate:.2f}%"}

    elif "highest booking cancellations" in query_lower:
        canceled_bookings = data[data['is_canceled'] == 1]
        highest_cancellations = canceled_bookings['country'].value_counts().idxmax()
        cancel_count = canceled_bookings['country'].value_counts().max()
        return {"Location with Highest Cancellations": highest_cancellations, "Total Cancellations": int(cancel_count)}

    elif "average price" in query_lower:
        avg_price = data[data['is_canceled'] == 0]['revenue'].mean()
        return {"Average Revenue per Booking": f"${avg_price:.2f}"}

    return None

def ask_question(query):
    # Try analytics query first
    analytics_response = handle_analytics_query(query)
    if analytics_response is not None:
        return analytics_response

    # Use RAG
    model = get_model()
    query_embedding = model.encode([query]).astype('float32')
    D, I = index.search(query_embedding, 5)
    context_data = "\n".join(data.iloc[I[0]]['text'].tolist())
    return {"answer": ask_gemini(query, context_data)}

# API Endpoints
@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    try:
        report = generate_analytics(request.report_type)
        return report
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@app.post("/ask")
async def answer_question(request: AskRequest):
    try:
        response = ask_question(request.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

