# Advanced Sentiment Analysis Platform

This project is a web-based NLP application that analyzes customer reviews and text data using Transformer models from Hugging Face.  
It can detect sentiment, emotions, generate summaries, and visualize text data through charts and word clouds.

The application is built using Streamlit and deployed online for public access.

## Live Demo

https://advanced-sentiment-analysis-platform-xfgaxmb6qeny2gvqvhltmb.streamlit.app/

---

## Features

- Analyze sentiment of text (Positive / Negative)
- Detect emotions from reviews
- Upload CSV files for bulk analysis
- Generate AI-based summaries
- Create word cloud visualizations
- Interactive charts and metrics
- Clean dark-themed dashboard UI

---

## Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- Pandas
- Matplotlib
- WordCloud

---

## Models Used

### Sentiment Analysis
`distilbert-base-uncased-finetuned-sst-2-english`

### Emotion Detection
`j-hartmann/emotion-english-distilroberta-base`

### Summary Generation
`GPT-2`

---

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/Sreeshant786/advanced-sentiment-analysis-platform.git
```

Move into the project folder:

```bash
cd advanced-sentiment-analysis-platform
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows
```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## Project Structure

```text
advanced-sentiment-analysis-platform/
│
├── app.py
├── requirements.txt
├── README.md
├── sample_reviews.csv
│
└── .streamlit/
    └── config.toml
```

---

## About the Project

I built this project to practice NLP, Transformers, and deployment of AI applications using Streamlit.  
The goal was to create a complete sentiment analysis dashboard that could handle both single text inputs and CSV-based review datasets.

This project helped me understand:
- Transformer pipelines
- NLP workflows
- Text visualization
- Streamlit dashboard development
- Deploying AI apps online

---

## Future Improvements

Some features I may add later:
- Twitter/X sentiment analysis
- Fake review detection
- Database integration
- User authentication
- Advanced analytics dashboard

---

## Author

Sreeshant Nair

GitHub: https://github.com/Sreeshant786
