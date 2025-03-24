# Building-an-LLM-Powered-Booking-Analytics-QA-System
A system for analyzing hotel booking data and answering questions using analytics and Retrieval-Augmented Generation (RAG). Built with Streamlit for an interactive UI, FastAPI for a REST API, and evaluated for accuracy and performance.

## Setup Instructions

### Prerequisites
- **Python**: 3.11+ (tested with 3.11.9)
- **Git**: For cloning the repository
- **Google Gemini API Key**: Required for RAG responses (set as an environment variable)

### Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/hotel-booking-analytics.git
   cd hotel-booking-analytics
   ```
2. **Create and Activate a Virtual Environment:**
   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   Linux/Mac:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set the Gemini API Key:**
   Windows:
   ```bash
   set GEMINI_API_KEY="your_api_key_here"
   ```
   Linux/Mac:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
5. **Obtain Large Files:**
   Large files (hotel_bookings_with_embeddings.csv, hotel_booking_index.faiss) are not included in this repository due to GitHub’s size limits.
   Option 1: Download from Google Drive: https://drive.google.com/drive/folders/1_BQRfm4tbmxmGg0X2kR2XQsWuM1f_ltV?usp=drive_link

   Option 2: Generate them by running:
   ```bash
   python Data_cleaning.py
   python embeddings_faiss.py
   ```
   Place these files in the project root directory (hotel-booking-analytics/).
### Running the System
   **Data Preparation:**
      **Cleaning and Visualization:**
         ```bash
         python Data_cleaning.py
         ```
      **Generate Embeddings and FAISS Index:**
         ```bash
         python embeddings_faiss.py
         ```
   **Streamlit App:**
      ```bash
      streamlit run app.py --server.fileWatcherType=none
      ```
      Access at: http://localhost:8501
   **FastAPI Server:**
      ```bash
      uvicorn api:app --reload
      ```
      Access Swagger UI at: http://127.0.0.1:8000/docs
   **Evaluation:**
      **Q&A Accuracy:**
         ```bash
         python evaluate_qa.py
         ```
      **Performance Benchmark:**
         ```bash
         python benchmark_api.py
         ```
## Sample Test Queries & Expected Answers

### API: POST /ask
   Query: {"question": "What’s the total revenue for July 2016?"}
      Expected: {"Total Revenue": "$1525019.05", "Month": "July", "Year": "2016"}
   Query: {"question": "What’s the cancellation rate?"}
      Expected: {"Cancellation Rate": "37.04%"}
   Query: {"question": "Which country has the highest booking cancellations?"}
      Expected: {"Location with Highest Cancellations": "PRT", "Total Cancellations": 27519}
   Query: {"question": "Why do people cancel bookings?"}
      Expected: {"answer": "Based on the limited data... potential factors include lead time, seasonality..."} (paraphrased; see full response in evaluation)
   Query: {"question": "What’s the busiest month for bookings?"}
      Expected: {"answer": "Based on limited data, June and December..."} (may vary with full dataset)

### API: POST /analytics
   Query: {"report_type": "revenue_trends"}
      Expected: List of {"arrival_date": "<date>", "revenue": <value>} (e.g., [{"arrival_date": "2015-07-01T00:00:00", "revenue": 123456.78}, ...])
   Query: {"report_type": "cancellation_rate"}
      Expected: {"cancellation_rate": "37.04%"}
   Query: {"report_type": "top_locations"}
      Expected: Dictionary of top 10 countries (e.g., {"PRT": 48590, "GBR": 12129, ...})

### Notes
   Data Source: Derived from a hotel bookings dataset (e.g., Kaggle). Raw data not included in GitHub; use Google Drive link or generate via scripts.
   Large Files: hotel_bookings_with_embeddings.csv and hotel_booking_index.faiss are available on Google Drive or can be regenerated.
   Evaluation Results:
   Accuracy: Quantitative 100%, Qualitative ~60-80% relevant.
   Latency: Analytics ~0.05s, RAG ~1s.
