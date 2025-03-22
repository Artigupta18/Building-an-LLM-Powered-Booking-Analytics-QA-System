#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# File: app.py
import streamlit as st
import pandas as pd
import faiss
import re
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure Gemini API (use environment variable or Streamlit secrets)
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "your_api_key"))  # Replace with your key
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Load data and FAISS index (no model yet)
data = pd.read_csv('hotel_bookings_with_embeddings.csv')
data['arrival_date'] = pd.to_datetime(data['arrival_date']) 
index = faiss.read_index('hotel_booking_index.faiss')

# Streamlit App Title
st.title("🏨 Hotel Booking Analytics & Q&A System")

# Sidebar Navigation
option = st.sidebar.selectbox("Choose an option:", ["Analytics", "Ask a Question"])

# Function to display analytics
def display_analytics():
    st.header("📊 Booking Analytics")

    # Revenue Trends
    st.subheader("💰 Revenue Trends Over Time")
    revenue_trends = data[data['is_canceled'] == 0].groupby(
        data['arrival_date'].dt.to_period('M')
    )['revenue'].sum().reset_index()
    revenue_trends['arrival_date'] = revenue_trends['arrival_date'].dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='arrival_date', y='revenue', data=revenue_trends, ax=ax)
    ax.set_title('Revenue Trends Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Revenue ($)')
    st.pyplot(fig)

    # Cancellation Rate
    st.subheader("❌ Cancellation Rate")
    cancellation_rate = data['is_canceled'].mean() * 100
    st.write(f"{cancellation_rate:.2f}% of total bookings were canceled.")

    # Geographical Distribution
    st.subheader("🌍 Top 10 Booking Locations")
    geo_distribution = data[data['is_canceled'] == 0]['country'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    geo_distribution.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Top 10 Booking Countries')
    ax.set_xlabel('Country')
    ax.set_ylabel('Number of Bookings')
    st.pyplot(fig)

    # Lead Time Distribution
    st.subheader("⏳ Booking Lead Time Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['lead_time'], bins=50, kde=True, ax=ax, color='orange')
    ax.set_title('Booking Lead Time Distribution')
    ax.set_xlabel('Lead Time (days)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Function to generate responses using Gemini
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

# Function to handle analytics queries with pandas
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

    elif "revenue trends" in query_lower:
        st.subheader("💰 Revenue Trends Over Time")
        revenue_trends = data[data['is_canceled'] == 0].groupby(
            data['arrival_date'].dt.to_period('M')
        )['revenue'].sum().reset_index()
        revenue_trends['arrival_date'] = revenue_trends['arrival_date'].dt.to_timestamp()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='arrival_date', y='revenue', data=revenue_trends, ax=ax)
        st.pyplot(fig)
        return None

    elif "geographical distribution" in query_lower:
        st.subheader("🌍 Top 10 Booking Locations")
        geo_distribution = data[data['is_canceled'] == 0]['country'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        geo_distribution.plot(kind='bar', ax=ax, color='skyblue')
        st.pyplot(fig)
        return None

    elif "lead time distribution" in query_lower:
        st.subheader("⏳ Booking Lead Time Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['lead_time'], bins=50, kde=True, ax=ax, color='orange')
        st.pyplot(fig)
        return None

    elif "average price" in query_lower:
        avg_price = data[data['is_canceled'] == 0]['revenue'].mean()
        return {"Average Revenue per Booking": f"${avg_price:.2f}"}

    return None

# Function to answer questions
def ask_question(query):
    # Try analytics query first
    analytics_response = handle_analytics_query(query)
    if analytics_response is not None:  # Handle both dict and None
        return analytics_response

    # Lazy-load sentence_transformers for RAG
    if not hasattr(st.session_state, 'model'):
        from sentence_transformers import SentenceTransformer
        st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
    model = st.session_state.model

    # Use FAISS for retrieval
    query_embedding = model.encode([query]).astype('float32')
    D, I = index.search(query_embedding, 5)
    context_data = "\n".join(data.iloc[I[0]]['text'].tolist())
    return {"Answer": ask_gemini(query, context_data)}

# App Logic
if option == "Analytics":
    display_analytics()

elif option == "Ask a Question":
    st.header("🧠 Ask a Booking-Related Question")
    # Add sample queries
    st.markdown("""
    **Not sure what to ask? Try these examples:**
    - "What are the revenue trends over time?" (Shows a chart)
    - "What’s the total revenue for July 2016?" (Shows revenue for a specific month)
    - "What’s the cancellation rate?" (Shows the overall cancellation percentage)
    - "Which country has the highest booking cancellations?" (Shows top cancellation location)
    - "Why do people cancel bookings?" (Explains possible reasons)
    - "What’s the busiest month for bookings?" (Identifies peak booking period)
    - "What’s the typical lead time for bookings from Portugal?" (Country-specific insight)
    """)
    query = st.text_input("Enter your question here:")
    if st.button("Ask"):
        if query:
            response = ask_question(query)
            if response:  
                st.subheader("🔍 Answer:")
                st.json(response)
            else:
                st.success("✅ Graph generated successfully.")
        else:
            st.warning("Please enter a question!")


# In[ ]:




