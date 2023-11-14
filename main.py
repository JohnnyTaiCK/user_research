import streamlit as st
import pandas as pd
import model
import openai
from bokeh.plotting import figure

openai.api_key = "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"
client = openai.OpenAI(api_key="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")

st.title("Recommendations for Amazon Sellers")

option = st.selectbox(
    "Which Model to use?",
    ("Default", "amazon-review-sentiment-analysis"),
    placeholder="Select a Model"
)


if option == "Default":
    df = model.defaultModel()
    Yearly_avg_rating = df.groupby("rev_year")["rating"].mean().reset_index()
    Yearly_avg_rating = Yearly_avg_rating.rename(columns = {"rating":"avg_rating"})
    # print(Yearly_avg_rating.head(10))

    chart_data = pd.DataFrame({
        "Year": Yearly_avg_rating['rev_year'],
        "Average Rating": Yearly_avg_rating['avg_rating']
    })

    container = st.container()
    container.empty()
    container.line_chart(chart_data, x="Year", y="Average Rating")
    recommendation = container.button("get Advice")
    grouped_df = df.groupby("rating_class")

    good_rev = grouped_df.get_group("good")["clean_summary"]
    good_rev = good_rev.astype("str")
    feq_good_rev = pd.Series(" ".join(good_rev).split()).value_counts()
    if recommendation:
        container.dataframe(feq_good_rev[0:12])
else:
    df = model.LiYuanModel()
    container = st.container()
    container.empty()
    A = df['overall'].tolist()
    B = df['label'].tolist()
    rating_level = [1, 2, 3, 4, 5]
    counts_A = [A.count(level) for level in rating_level]
    counts_B = [B.count(level) for level in rating_level]
    chart_data = pd.DataFrame({
        "review-rating": counts_A,
        "text-rating": counts_B,
    })
    container.bar_chart(chart_data)