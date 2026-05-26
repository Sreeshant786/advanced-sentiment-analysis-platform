import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Advanced Sentiment Analysis Platform",
    layout="wide"
)

# --------------------------------
# TITLE
# --------------------------------

st.title("Advanced Sentiment Analysis Platform")

st.markdown("---")

# --------------------------------
# LOAD MODELS
# --------------------------------

@st.cache_resource
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_emotion_model():

    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )

@st.cache_resource
def load_summary_model():

    return pipeline(
        "text-generation",
        model="gpt2"
    )

sentiment_classifier = load_sentiment_model()

emotion_classifier = load_emotion_model()

summary_classifier = load_summary_model()

# --------------------------------
# SINGLE TEXT ANALYSIS
# --------------------------------

st.header("Single Text Analysis")

text = st.text_area(
    "Enter text",
    height=150
)

if st.button("Analyze Text"):

    if text.strip():

        # -------------------------
        # SENTIMENT
        # -------------------------

        sentiment_result = (
            sentiment_classifier(text)
        )

        sentiment_label = (
            sentiment_result[0]["label"]
        )

        sentiment_score = (
            sentiment_result[0]["score"]
        )

        # -------------------------
        # EMOTION
        # -------------------------

        emotion_result = (
            emotion_classifier(text)
        )

        emotion_label = (
            emotion_result[0][0]["label"]
        )

        emotion_score = (
            emotion_result[0][0]["score"]
        )

        # -------------------------
        # DISPLAY
        # -------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Sentiment")

            if sentiment_label == "POSITIVE":

                st.success(
                    f"{sentiment_label}"
                )

            else:

                st.error(
                    f"{sentiment_label}"
                )

            st.write(
                f"Confidence: {sentiment_score:.2f}"
            )

            st.subheader("Emotion")

            st.info(
                f"{emotion_label}"
            )

            st.write(
                f"Emotion Score: {emotion_score:.2f}"
            )

        with col2:

            fig, ax = plt.subplots()

            labels = [
                "Sentiment",
                "Emotion"
            ]

            scores = [
                sentiment_score,
                emotion_score
            ]

            ax.bar(
                labels,
                scores
            )

            ax.set_ylim(0, 1)

            st.pyplot(fig)

    else:

        st.warning(
            "Please enter text"
        )

# --------------------------------
# CSV ANALYSIS
# --------------------------------

st.markdown("---")

st.header("CSV Review Analysis")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # -------------------------
        # READ CSV
        # -------------------------

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.write(df.head())

        # -------------------------
        # SELECT COLUMN
        # -------------------------

        column = st.selectbox(
            "Select text column",
            df.columns
        )

        # -------------------------
        # ANALYZE BUTTON
        # -------------------------

        if st.button("Analyze CSV"):

            sentiments = []
            sentiment_scores = []

            emotions = []
            emotion_scores = []

            reviews = (
                df[column]
                .fillna("")
                .astype(str)
                .tolist()
            )

            # -------------------------
            # ANALYZE REVIEWS
            # -------------------------

            with st.spinner(
                "Analyzing reviews..."
            ):

                for review in reviews:

                    review = review.strip()

                    if review:

                        # SENTIMENT

                        s_result = (
                            sentiment_classifier(
                                review
                            )
                        )

                        sentiments.append(
                            s_result[0]["label"]
                        )

                        sentiment_scores.append(
                            s_result[0]["score"]
                        )

                        # EMOTION

                        e_result = (
                            emotion_classifier(
                                review
                            )
                        )

                        emotions.append(
                            e_result[0][0]["label"]
                        )

                        emotion_scores.append(
                            e_result[0][0]["score"]
                        )

                    else:

                        sentiments.append(
                            "EMPTY"
                        )

                        sentiment_scores.append(
                            0
                        )

                        emotions.append(
                            "EMPTY"
                        )

                        emotion_scores.append(
                            0
                        )

            # -------------------------
            # SAVE RESULTS
            # -------------------------

            df["Sentiment"] = sentiments

            df["Sentiment Score"] = (
                sentiment_scores
            )

            df["Emotion"] = emotions

            df["Emotion Score"] = (
                emotion_scores
            )

            # -------------------------
            # SHOW RESULTS
            # -------------------------

            st.subheader(
                "Analysis Results"
            )

            st.write(df)

            # -------------------------
            # METRICS
            # -------------------------

            positive_count = (
                df["Sentiment"]
                == "POSITIVE"
            ).sum()

            negative_count = (
                df["Sentiment"]
                == "NEGATIVE"
            ).sum()

            empty_count = (
                df["Sentiment"]
                == "EMPTY"
            ).sum()

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Positive Reviews",
                    positive_count
                )

            with col2:

                st.metric(
                    "Negative Reviews",
                    negative_count
                )

            with col3:

                st.metric(
                    "Empty Reviews",
                    empty_count
                )

            # -------------------------
            # PIE CHART
            # -------------------------

            st.subheader(
                "Sentiment Distribution"
            )

            sentiment_counts = (
                df["Sentiment"]
                .value_counts()
            )

            fig2, ax2 = plt.subplots()

            ax2.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct="%1.1f%%"
            )

            st.pyplot(fig2)

            # -------------------------
            # EMOTION CHART
            # -------------------------

            st.subheader(
                "Emotion Distribution"
            )

            emotion_counts = (
                df["Emotion"]
                .value_counts()
            )

            fig4, ax4 = plt.subplots()

            ax4.bar(
                emotion_counts.index,
                emotion_counts.values
            )

            st.pyplot(fig4)

            # -------------------------
            # WORD CLOUD
            # -------------------------

            st.subheader(
                "Word Cloud"
            )

            valid_reviews = []

            for review in reviews:

                review = str(
                    review
                ).strip()

                if len(review) > 2:

                    valid_reviews.append(
                        review
                    )

            all_text = " ".join(
                valid_reviews
            ).strip()

            if (
                all_text
                and
                len(all_text.split()) > 0
            ):

                try:

                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        min_word_length=1
                    ).generate(all_text)

                    fig3, ax3 = plt.subplots(
                        figsize=(10, 5)
                    )

                    ax3.imshow(
                        wordcloud,
                        interpolation="bilinear"
                    )

                    ax3.axis("off")

                    st.pyplot(fig3)

                except Exception as wc_error:

                    st.warning(
                        f"Word cloud error: {wc_error}"
                    )

            else:

                st.warning(
                    "No valid text available for word cloud"
                )

            # -------------------------
            # AI GENERATED SUMMARY
            # -------------------------

            st.subheader(
                "AI Generated Summary"
            )

            try:

                combined_text = " ".join(
                    valid_reviews
                )

                combined_text = (
                    combined_text[:500]
                )

                if (
                    len(
                        combined_text.split()
                    ) > 20
                ):

                    prompt = f"""
                    Summarize these customer reviews:

                    {combined_text}

                    Summary:
                    """

                    summary = (
                        summary_classifier(
                            prompt,
                            max_new_tokens=60,
                            do_sample=False
                        )
                    )

                    generated_text = (
                        summary[0]["generated_text"]
                    )

                    st.success(
                        generated_text
                    )

                else:

                    st.warning(
                        "Not enough text for summary generation"
                    )

            except Exception as summary_error:

                st.warning(
                    f"Summary generation error: {summary_error}"
                )

    except Exception as e:

        st.error(f"Error: {e}")