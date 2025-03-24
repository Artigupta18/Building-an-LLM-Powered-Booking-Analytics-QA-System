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
```
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
```
set GEMINI_API_KEY="your_api_key_here"
```
Linux/Mac:
```bash
export GEMINI_API_KEY="your_api_key_here"
```
Obtain Large Files:
Large files (hotel_bookings_with_embeddings.csv, hotel_booking_index.faiss) are not included in this repository due to GitHub’s size limits.
Option 1: Download from Google Drive: https://drive.google.com/drive/folders/your-drive-folder-id
Option 2: Generate them by running:
bash

Collapse

Wrap

Copy
python Data_cleaning.py
python embeddings_faiss.py
Place these files in the project root directory (hotel-booking-analytics/).
