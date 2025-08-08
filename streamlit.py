import streamlit as st
import requests
import json

# --- Configuration ---
# Ensure your FastAPI backend is running on this URL
FASTAPI_URL = "http://127.0.0.1:6006" # <--- CHANGED THIS PORT TO 6006
GENERATE_ENDPOINT = f"{FASTAPI_URL}/ai/generate-and-review"

# --- Streamlit UI ---
st.set_page_config(page_title="SensAI Course Creator", layout="wide")

st.title("📚 SensAI AI Course Creator Assistant")
st.markdown("Enter a topic below, and let the AI draft a full course outline, lesson, and questions, then review it.")

# Input for the topic
topic_input = st.text_input("Course Topic", placeholder="e.g., Introduction to Python Programming")

# Button to trigger generation
if st.button("Generate & Review Course"):
    if topic_input:
        st.info("Generating course content and running AI review... This may take a moment.")
        
        try:
            # Prepare the request payload
            payload = {"topic": topic_input}
            
            # Make the POST request to your FastAPI backend
            response = requests.post(GENERATE_ENDPOINT, json=payload)
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                st.success("Course content generated and reviewed successfully!")
                
                # --- Display Results ---
                st.subheader("Generated Course Draft")
                st.json(result['draft']) # Display the entire draft object as JSON

                st.subheader("AI Review Feedback")
                st.json(result['review']) # Display the review feedback

                st.subheader("Publish Status")
                st.json(result['publish_status']) # Display the publish status

            else:
                st.error(f"Error generating course: HTTP Status {response.status_code}")
                st.json(response.json()) # Display error details from backend
        
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to FastAPI backend at {FASTAPI_URL}. Please ensure it's running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a topic to generate the course content.")

st.markdown("---")
st.markdown("Powered by FastAPI & Streamlit")