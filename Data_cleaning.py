#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("hotel_bookings.csv")


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[6]:


# Handle missing values (avoid inplace=True)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)
df['children'] = df['children'].fillna(0)  # Assume 0 children if missing
df['country'] = df['country'].fillna('Unknown') 


# In[7]:


# Calculate total nights and revenue
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['revenue'] = df['adr'] * df['total_nights']

# Create a unified date column for easier analysis
df['arrival_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' + 
    df['arrival_date_month'] + '-' + 
    df['arrival_date_day_of_month'].astype(str),
    format='%Y-%B-%d'
)

# Drop unnecessary columns (optional, adjust as needed)
df = df.drop(columns=['reservation_status_date', 'agent', 'company'])

# Save cleaned data
df.to_csv("hotel_bookings_cleaned.csv", index=False)
print("Data cleaning completed. Saved to 'hotel_bookings_cleaned.csv'.")


# # Data Visualisation

# In[8]:


# File: data_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("hotel_bookings_cleaned.csv")


# In[10]:


# Convert arrival_date to datetime (if not already)
df['arrival_date'] = pd.to_datetime(df['arrival_date'])


# In[11]:


# 1. Revenue trends over time (non-canceled bookings only)
revenue_trends = df[df['is_canceled'] == 0].groupby(
    df['arrival_date'].dt.to_period('M')
)['revenue'].sum().reset_index()
revenue_trends['arrival_date'] = revenue_trends['arrival_date'].dt.to_timestamp()

# 2. Cancellation rate
cancellation_rate = df['is_canceled'].mean() * 100

# 3. Geographical distribution (non-canceled bookings)
geo_distribution = df[df['is_canceled'] == 0]['country'].value_counts().head(10)

# 4. Lead time distribution
lead_time_distribution = df['lead_time']

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Revenue trends
sns.lineplot(x='arrival_date', y='revenue', data=revenue_trends, ax=axes[0, 0])
axes[0, 0].set_title('Revenue Trends Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Total Revenue ($)')

# Cancellation rate
axes[0, 1].text(0.5, 0.5, f'Cancellation Rate: {cancellation_rate:.2f}%', 
                ha='center', va='center', fontsize=14, bbox=dict(facecolor='lightgrey', alpha=0.5))
axes[0, 1].set_axis_off()

# Geographical distribution
geo_distribution.plot(kind='bar', ax=axes[1, 0], color='skyblue')
axes[1, 0].set_title('Top 10 Booking Countries')
axes[1, 0].set_xlabel('Country')
axes[1, 0].set_ylabel('Number of Bookings')

# Lead time distribution
sns.histplot(lead_time_distribution, bins=50, kde=True, ax=axes[1, 1], color='orange')
axes[1, 1].set_title('Booking Lead Time Distribution')
axes[1, 1].set_xlabel('Lead Time (days)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("booking_visualizations.png")  # Save for reference
plt.show()
print("Visualizations generated and saved as 'booking_visualizations.png'.")

