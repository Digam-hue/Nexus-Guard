#!/bin/bash

# Start the FastAPI server in the background
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --workers 1 &

# Wait a bit for the server to start up
sleep 10

# Start the Streamlit app
streamlit run app/streamlit_app.py --server.port $PORT --server.enableCORS false