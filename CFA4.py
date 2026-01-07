import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
import re
import time  # For animations
from collections import Counter
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind
from datetime import datetime
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data (only need to run once)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Load data and perform preprocessing
@st.cache_data
def load_and_preprocess_data():
    result = pd.read_csv("amazon_review.csv")
    result.drop(columns={"asin", "helpful", "unixReviewTime", "helpful_yes", "total_vote"}, inplace=True)
    result = result.dropna()
    result.columns = result.columns.str.strip()

     # Encoding 'reviewerName' to use it as a feature in Time Series model
    label_encoder = LabelEncoder()
    result['reviewerName_encoded'] = label_encoder.fit_transform(result['reviewerName'])

     # Convert reviewTime to datetime
    result['reviewTime'] = pd.to_datetime(result['reviewTime'], dayfirst=True)

    # -- Calculate review text length
    result['reviewTextLength'] = result['reviewText'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)

    # -- Sentiment analysis using VADER
    sia = SentimentIntensityAnalyzer()
    result["sentiment_score"] = result["reviewText"].apply(lambda text: sia.polarity_scores(text)['compound'] if isinstance(text, str) else 0)

    # Add a product category column (replace with your actual logic to determine category)
    # This is just a placeholder, adapt based on your dataset or logic

    # Create product_id (for easier review assignment)
    num_products = 40  # Increased for new categories
    result['product_id'] = [random.randint(1, num_products) for _ in range(len(result))]

    # Calculate day_diff: difference in days between the review date and the latest date in the dataset
    result['day_diff'] = (result['reviewTime'].max() - result['reviewTime']).dt.days

    return result


result = load_and_preprocess_data()

# --- Sidebar for Navigation ---
st.sidebar.title("Amazon Review ")  # Added the main title at the top of sidebar
st.sidebar.header("Explore:")  # Added the Navigation heading
menu_options = [
    "Home",
    "Product Reviews",
    "Random Forest Regression",
    "Clustering Analysis",
    "XGBoost Regression",
    "RFMT Model",
    "Data Visualization",
    "Sentiment Analysis",
    "Data Exploration",
    "Feedback Rating"
]

# Define a dictionary for coloring the side headings
menu_colors = {
    "Product Reviews": "red",
    "Random Forest Regression": "green",
    "Clustering Analysis": "blue",
    "XGBoost Regression": "orange",
    "RFMT Model": "purple",
    "Data Visualization": "teal",
    "Sentiment Analysis": "pink",
    "Data Exploration": "brown",
    "Feedback Rating": "gray"
}

# Custom CSS to color the sidebar options
custom_css = """
<style>
    [data-baseweb="select"] div > div {
    background-color: #ffffff !important; /* Change background color */
    }
    .sidebar .sidebar-content {{
        background-color: #f0f2f6; /* Optional: Change sidebar background */
    }}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state for selected_menu if it doesn't exist
if 'selected_menu' not in st.session_state:
    st.session_state.selected_menu = "Home"

selected_menu = st.sidebar.selectbox("Main Sections", menu_options, index=menu_options.index(st.session_state.selected_menu))


# --- Main Content Based on Selection ---
if selected_menu == "Home":
    st.title("Amazon Review Analysis")

    st.markdown("""
    <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; border: 2px solid black;">
        <h2>Welcome!</h2>
        <p>This application provides a comprehensive toolkit for analyzing Amazon product reviews.
        Gain valuable insights into customer sentiment, predict product ratings using machine learning, and
        explore the underlying data trends.</p>
        <p>Uncover hidden patterns in customer feedback and make data-driven decisions to improve product offerings and customer satisfaction.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Key Features")
    st.markdown("""
    <ul>
        <li><b style="color: red;">Product Review Exploration:</b> Browse real-world reviews categorized by product type.</li>
        <li><b style="color: red;">Sentiment Analysis:</b> Understand the emotional tone of customer reviews.</li>
        <li><b style="color: red;">Predictive Modeling:</b> Employ machine learning models like Random Forest and XGBoost to predict product ratings.</li>
        <li><b style="color: red;">Clustering Analysis:</b> Identify distinct customer segments based on review characteristics.</li>
        <li><b style="color: red;">Data Visualization:</b> Create insightful charts to visualize review trends and patterns.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.subheader("Get Started")
    st.write("Use the sidebar on the left to navigate to different sections of the app. Explore product reviews, run predictive models, analyze sentiment, and visualize the data.")

    # Add button to navigate to Product Reviews page
    if st.button("Reviews"):
        st.session_state.selected_menu = "Product Reviews"
        st.rerun()

# --- Product Reviews Section ---
elif selected_menu == "Product Reviews":
    st.header("Product Reviews by Category")

    # Define product categories with placeholder images
    categories = {
        "Books": {
            "image": "images/book.jpg",
            "products": [
                {"name": "Biography", "image": "images/biography.jpg", "product_id": 1},
                {"name": "Self-Help Book", "image": "images/self-help.jpg", "product_id": 2},
                {"name": "Cookbook", "image": "images/cookbook.jpg", "product_id": 3},
                {"name": "Thriller", "image": "images/thriller.jpg", "product_id": 4},
                {"name": "Historical", "image": "images/historical.jpg", "product_id": 5},
                {"name": "Romantic", "image": "images/romantic.jpg", "product_id": 6},
                {"name": "Epic", "image": "images/epic.jpg", "product_id": 7},
                {"name": "Novel", "image": "images/novel.jpg", "product_id": 8},
                {"name": "Finance", "image": "images/finance.jpg", "product_id": 9},
                {"name": "Jungle", "image": "images/jungle.jpg", "product_id": 10},
            ]
        },
        "Electronics": {
            "image": "images/Electronics.jpg",
            "products": [
                {"name": "Wireless Earbuds", "image": "images/wirless buds.jpg", "product_id": 11},
                {"name": "Smartwatch", "image": "images/smartwatch.jpg", "product_id": 12},
                {"name": "Laptop", "image": "images/laptop image.jpg", "product_id": 13},
                {"name": "HomeTheater", "image": "images/hometheater.jpeg", "product_id": 14},
                {"name": "Amazon Firestick", "image": "images/amazon firestick.jpg", "product_id": 15},
                {"name": "Ipad", "image": "images/ipad.jpg", "product_id": 16},
                {"name": "Ac", "image": "images/ac.jpg", "product_id": 17},
                {"name": "Ps5", "image": "images/ps5.jpeg", "product_id": 18},
                {"name": "TV", "image": "images/tv.jpg", "product_id": 19},
                {"name": "Phone", "image": "images/phone.jpg", "product_id": 20},
            ]
        },
        "Home & Kitchen": {
            "image": "images/home.jpg",
            "products": [
                {"name": "Blender", "image": "images/blender.jpg", "product_id": 21},
                {"name": "Air Fryer", "image": "images/air fryer.jpg", "product_id": 22},
                {"name": "Coffee Maker", "image": "images/coffee makers.jpg", "product_id": 23},
                {"name": "Toaster", "image": "images/toaster.jpeg", "product_id": 24},
                {"name": "Washing Machine", "image": "images/washing machine.jpg", "product_id": 25},
                {"name": "Pressure Cooker", "image": "images/pressure cooker.jpg", "product_id": 26},
                {"name": "Electric stove", "image": "images/electric stove.jpg", "product_id": 27},
                {"name": "Refrigerator", "image": "images/refrigerator.jpg", "product_id": 28},
                {"name": "Microwave", "image": "images/microwave.jpg", "product_id": 29},
                {"name": "Dishwasher", "image": "images/dishwasher.jpg", "product_id": 30},
            ]
        },
        "Clothing": {
            "image": "images/clothes.jpeg",
            "products": [
                {"name": "T-Shirt", "image": "images/tshirt.jpeg", "product_id": 31},
                {"name": "Jeans", "image": "images/jeans.jpeg", "product_id": 32},
                {"name": "Jacket", "image": "images/jacket.jpg", "product_id": 33},
                {"name": "Dress", "image": "images/dress.jpg", "product_id": 34},
                {"name": "Shoes", "image": "images/shoes.jpg", "product_id": 35},
                {"name": "Sweater", "image": "images/sweater.jpg", "product_id": 36},
                {"name": "Hat", "image": "images/hat.jpg", "product_id": 37},
                {"name": "Socks", "image": "images/socks.jpeg", "product_id": 38},
                {"name": "Pants", "image": "images/pants.jpeg", "product_id": 39},
                {"name": "Skirt", "image": "images/skrit.jpeg", "product_id": 40},
            ]
        },
    }

    # --- Session State Management ---
    if 'selected_product_id' not in st.session_state:
        st.session_state.selected_product_id = None
        st.session_state.reviews_displayed = False

    def display_reviews(product_id, product_name):
        st.session_state.selected_product_id = product_id
        st.session_state.reviews_displayed = True
        st.session_state.product_name = product_name # Store product name

    def clear_selection():
        st.session_state.selected_product_id = None
        st.session_state.reviews_displayed = False
        st.session_state.product_name = None

    # --- Display Product Reviews or Categories ---
    if st.session_state.selected_product_id:
        # Retrieve reviews for the selected product using product_id
        product_reviews = result[result['product_id'] == st.session_state.selected_product_id]
        st.subheader(f"Reviews for {st.session_state.product_name}")
        st.dataframe(product_reviews)  # Display reviews

        if st.button("Back to Categories"):
            clear_selection()

    else:
        # Display product categories and images if no product is selected
        selected_category = st.selectbox("Select a Category", list(categories.keys()))

        # Show category image
        st.image(categories[selected_category]["image"], caption=selected_category, width=700)

        # Display products in the selected category
        st.subheader(f"Products in {selected_category}")
        cols = st.columns(5)  # Create 5 columns for the grid

        for idx, product in enumerate(categories[selected_category]["products"]):
            with cols[idx % 5]:  # Distribute images across columns
                st.image(product["image"], caption=product["name"], width=140)  # Display product image

                if st.button(f"Reviews for {product['name']}", key=f"reviews_{product['name']}"):  # Add a button for each product
                    display_reviews(product['product_id'], product['name'])

# --- Random Forest Regression Section ---
elif selected_menu == "Random Forest Regression":
    st.header("Random Forest Regression Model")
    # Using 'overall' as the target for demonstration, change this as needed.
    X_reg = result[["day_diff"]]
    y_reg = result["overall"]

    # Check for data availability
    if X_reg.empty or y_reg.empty:
        st.error("Error: Unable to perform random forest regression. Please ensure data is loaded.")
    else:
        # Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

        # Train Random Forest model
        with st.spinner("Training Random Forest model..."):  # Add a spinner while training
            time.sleep(1)
            model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Round the predicted values to the nearest integer if predicting ratings
        y_pred_rounded = np.round(y_pred)


        # Display MSE and R2 score
        st.subheader("Model Evaluation")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
            # Plotting the data points
        st.subheader("Scatter Plot of Training Data")
        fig_scatter, ax_scatter = plt.subplots()
        ax_scatter.scatter(X_train, y_train, label="Training Data Points", s=5)
        ax_scatter.set_xlabel("Day Diff")
        ax_scatter.set_ylabel("Overall Rating")
        ax_scatter.legend()
        st.pyplot(fig_scatter)

# --- Clustering Section ---
elif selected_menu == "Clustering Analysis":
    st.header("Clustering Analysis")

    # Select data for clustering (example: day_diff and overall rating)
    X_cluster = result[["day_diff", "overall"]].copy()

    if X_cluster.empty:
        st.error("Error: Unable to perform clustering. Please ensure data is loaded.")
    else:
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        # Clustering Algorithm Selection
        cluster_alg = st.selectbox("Select a Clustering Algorithm",
                                  ["K-Means", "Gaussian Mixture", "DBSCAN", "Agglomerative"])

        if cluster_alg == "K-Means":
            #n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
            with st.spinner("Performing K-Means clustering..."):  # Add a spinner while clustering
                    time.sleep(1)
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')  # Default n_clusters to 3
                    clusters = kmeans.fit_predict(X_scaled)
        elif cluster_alg == "Gaussian Mixture":
            #n_components = st.slider("Number of Components", min_value=2, max_value=10, value=3)
            with st.spinner("Performing Gaussian Mixture clustering..."):  # Add a spinner while clustering
                    time.sleep(1)
                    gmm = GaussianMixture(n_components=3, random_state=42)  # Default n_components to 3
                    clusters = gmm.fit_predict(X_scaled)
        elif cluster_alg == "DBSCAN":
            eps = st.slider("Epsilon", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            min_samples = st.slider("Minimum Samples", min_value=2, max_value=20, value=5)
            with st.spinner("Performing DBSCAN clustering..."):  # Add a spinner while clustering
                    time.sleep(1)
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(X_scaled)
        elif cluster_alg == "Agglomerative":
            #n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
            with st.spinner("Performing Agglomerative clustering..."):  # Add a spinner while clustering
                time.sleep(1)
                agg = AgglomerativeClustering(n_clusters=3) # Default n_clusters to 3
                clusters = agg.fit_predict(X_scaled)


        # Visualize the clustering results
        st.subheader(f"{cluster_alg}")

        # Use Plotly Express for interactive 3D scatter plot
        cluster_df = pd.DataFrame(X_scaled, columns=["Scaled Day Diff", "Scaled Overall Rating"])
        cluster_df["Cluster"] = clusters  # Add cluster labels to dataframe
        cluster_df["reviewTextLength"] = result["reviewTextLength"]  # Add a third dimension

        fig_cluster = px.scatter_3d(cluster_df,
                                  x="Scaled Day Diff",
                                  y="Scaled Overall Rating",
                                  z="reviewTextLength",  # Use reviewTextLength as the third dimension
                                  color="Cluster",
                                  hover_data=["Scaled Day Diff", "Scaled Overall Rating", "Cluster", "reviewTextLength"],
                                  title=f"{cluster_alg} Clustering Results")

        st.plotly_chart(fig_cluster)

# --- XGBoost Regression Section ---
elif selected_menu == "XGBoost Regression":
    st.header("XGBoost Regression Model")
    X_xgb = result[["day_diff"]]  # Features for XGBoost
    y_xgb = result["overall"]  # target variable\

    if X_xgb.empty or y_xgb.empty:
        st.error("Error: Unable to perform XGBoost regression. Please ensure data is loaded.")
    else:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.3, random_state=42)

        # Train XGBoost model
        with st.spinner("Training XGBoost model..."):  # Add a spinner while training
            time.sleep(1)
            xgbr = xgb.XGBRegressor(objective='reg:squarederror',
                                n_estimators=5,
                                seed=42)  # other model parameters can be adjusted here
            xgbr.fit(X_train, y_train)

        # Make predictions
        y_pred = xgbr.predict(X_test)
        y_train_pred = xgbr.predict(X_train)  # Predict on training data

        # --- Round predictions to the nearest integer if predicting ratings
        y_pred = np.round(y_pred)
        y_train_pred = np.round(y_train_pred)


        # Calculate accuracy percentage
        train_accuracy = np.mean(y_train == y_train_pred) * 100
        test_accuracy = np.mean(y_test == y_pred) * 100



        # Evaluate the model
        st.subheader("Model Evaluation")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error (Test): {mse:.2f}")
        st.write(f"R2 Score (Test): {r2:.2f}")
        st.write(f"Training Accuracy: {train_accuracy:.2f}%")  # Display training accuracy
        st.write(f"Testing Accuracy: {test_accuracy:.2f}%")    # Display test accuracy

        # Scatter Plot of the results
        st.subheader("Scatter Plot of XGBoost Predictions")
        fig_xgb, ax_xgb = plt.subplots()
        ax_xgb.scatter(X_test.iloc[:, 0], y_test, label="Actual Values", color='blue', s=5)  # Scatterplot of feature day_diff vs actual overall rating
        ax_xgb.scatter(X_test.iloc[:, 0], y_pred, label="Predicted Values", color='green', s=5)  # Scatterplot of feature day_diff vs predicted overall rating
        ax_xgb.set_xlabel("Day Diff")
        ax_xgb.set_ylabel("Overall Rating")
        ax_xgb.legend()
        st.pyplot(fig_xgb)


# --- RFMT Model Section ---
elif selected_menu == "RFMT Model":
    st.header("Recency, Frequency, Monetary,Time (RFMT)")

    # Define features (use reviewerName_encoded)
    st.subheader("Hyperparameter Tuning")
    n_estimators = st.slider("Select Number of Estimators", min_value=50, max_value=500, value = 100)
    X_rfmt = result[['day_diff', 'reviewerName_encoded','reviewTextLength','sentiment_score']]  # Includes reviewer encoded for time series
    y_rfmt = result['overall']

    if X_rfmt.empty or y_rfmt.empty:
        st.error("Error: Unable to perform RFMT regression. Please ensure data is loaded.")
    else:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_rfmt, y_rfmt, test_size=0.3, random_state=42)
            # RFMT Model Training and Prediction
            with st.spinner("Training RFMT Model..."):
                time.sleep(1)
                rfmt_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                rfmt_model.fit(X_train, y_train)
                y_pred_rfmt = rfmt_model.predict(X_test)

            # Model Evaluation
            st.subheader("RFMT Model Evaluation")

            # -- Calculate R (R-squared) value
            r_value = r2_score(y_test, y_pred_rfmt)

            # Recency: The most recent review date.  This part is critically flawed and requires fixing.
            # Frequency:  This should relate to how often *each* reviewer appears.
            # Monetary: Sum of all the reviews is generally nonsensical for RFM.

            # Recalculate max_date and recency here, using only data in 'result' dataframe
            max_date = result['reviewTime'].max()
            min_date = result['reviewTime'].min()  # Get the earliest date

            # Time range in days
            time_range = (max_date - min_date).days

            # Recency Calculation: Days since the last review
            latest_date = result['reviewTime'].max()
            result['days_since_review'] = (latest_date - result['reviewTime']).dt.days
            recency = result['days_since_review'].min()

            # Frequency Calculation: Number of reviews per customer
            frequency = result['reviewerName'].value_counts().mean()

            # Monetary Value: Average rating given by a reviewer
            monetary = result.groupby('reviewerName')['overall'].mean().mean()


            # Normalization to Percentages
            # Ensure min_date is defined before using it
            if 'reviewTime' in result.columns:  # Check if column exists

                # Recency percentage (lower is better) - INVERTED
                recency_percentage = ((time_range - recency) / time_range) * 100 if time_range > 0 else 0

                # Frequency percentage
                max_frequency = result['reviewerName'].value_counts().max()
                frequency_percentage = (frequency / max_frequency) * 100 if max_frequency > 0 else 0

                # Monetary percentage
                max_monetary = result['overall'].max() # This assumes max possible rating is highest value
                monetary_percentage = (monetary / max_monetary) * 100 if max_monetary > 0 else 0  # Fixed ZeroDivisionError
            else:
                recency_percentage = 0
                frequency_percentage = 0
                monetary_percentage = 0


            # -- Display calculated values
            st.write(f"R-Squared (R Value): {r_value:.2f}")
            st.write(f"Recency (Percentage): {recency_percentage:.2f}%")
            st.write(f"Frequency (Percentage): {frequency_percentage:.2f}%")
            st.write(f"Monetary (Percentage): {monetary_percentage:.2f}%")
            st.write(f"Time Range (days): {time_range}")

            # Display MSE and R2 score
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rfmt):.2f}")
            st.write(f"R2 Score: {r2_score(y_test, y_pred_rfmt):.2f}")

            # Plotting the results
            st.subheader("Scatter Plot of RFMT Predictions")
            fig_rfmt, ax_rfmt = plt.subplots()
            ax_rfmt.scatter(X_test.iloc[:, 0], y_test, label="Actual Values", color='blue', s=5)
            ax_rfmt.scatter(X_test.iloc[:, 0], y_pred_rfmt, label="Predicted Values", color='red', s=5)
            ax_rfmt.set_xlabel("Day Diff")
            ax_rfmt.set_ylabel("Overall Rating")
            ax_rfmt.legend()
            st.pyplot(fig_rfmt)

            # --- Bar Chart (Feature Importances) ---
            st.subheader("Feature Importances")
            importances = rfmt_model.feature_importances_
            feature_names = X_rfmt.columns
            fig_bar_imp, ax_bar_imp = plt.subplots()
            ax_bar_imp.bar(feature_names, importances)
            ax_bar_imp.tick_params(axis='x', rotation=45)
            ax_bar_imp.set_xlabel('Features')
            ax_bar_imp.set_ylabel('Importance')
            st.pyplot(fig_bar_imp)

            # --- Pie Chart (RFMT Performance) ---
            st.subheader("RFMT Performance Pie Chart")
            performance_metrics = {
                'Recency': recency_percentage,
                'Frequency': frequency_percentage,
                'Monetary': monetary_percentage
            }

            labels = list(performance_metrics.keys())
            values = list(performance_metrics.values())

            fig_pie_perf, ax_pie_perf = plt.subplots()
            ax_pie_perf.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
            st.pyplot(fig_pie_perf)


# --- Data Visualization ---
elif selected_menu == "Data Visualization":
    st.header("Data Visualization")
    # --- Pie Chart 1 (Reviewer vs Overall) ---
    st.subheader("Pie Chart: Reviewer Count vs. Overall Rating Sum")
    x = result['reviewerName'].count()
    y = result['overall'].sum()
    sizes = [x, y]
    labels = ['reviewerName Count', 'Overall Rating Sum']

    fig_pie1, ax_pie1 = plt.subplots()
    ax_pie1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['blue', 'green'])
    st.pyplot(fig_pie1)

    # --- Pie Chart 2 (Rating Distribution) ---
    st.subheader("Pie Chart: Distribution of Overall Ratings")
    rating_counts = result['overall'].value_counts().sort_index()
    fig_pie2, ax_pie2 = plt.subplots()
    colors = ['red', 'orange', 'yellow', 'skyblue', 'grey']
    ax_pie2.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    st.pyplot(fig_pie2)

    # ---  Bar Chart  (Reviewer vs Rating Sum) ---
    st.subheader("Bar Chart : Reviewer Vs. Rating Sum")
    reviewer_rating_sum = result.groupby('reviewerName')['overall'].sum().sort_values(ascending=False).head(10)  # Top 10 reviewers
    fig_bar = px.bar(x=reviewer_rating_sum.index, y=reviewer_rating_sum.values,
                    labels={'x': 'Reviewer Name', 'y': 'Total Rating Sum'})
    st.plotly_chart(fig_bar)

# --- Sentiment Analysis ---
elif selected_menu == "Sentiment Analysis":
    st.header("Sentiment Analysis of Review Text")

    if st.checkbox("Analyze Sentiment"):

        # Use VADER for sentiment analysis
        sia = SentimentIntensityAnalyzer()

        def analyze_sentiment(text):
            # Get polarity scores from VADER
            if not isinstance(text, str):
                return "N/A"
            scores = sia.polarity_scores(text)
            compound_score = scores['compound']  # Overall sentiment score

            # Determine sentiment label based on compound score
            if compound_score >= 0.05:
                return "Positive"
            elif compound_score <= -0.05:
                return "Negative"
            else:
                return "Neutral"

        with st.spinner("Analyzing review sentiments..."):
            time.sleep(1)
            result["sentiment"] = result["reviewText"].apply(analyze_sentiment)

            sentiment_counts = result["sentiment"].value_counts()

            st.subheader("Review Sentiment Distribution")
            st.write(sentiment_counts)

            fig_sent, ax_sent = plt.subplots()
            ax_sent.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140,
                        colors=["lightgreen", "lightcoral", "skyblue"])
            st.pyplot(fig_sent)

            # Generate Word Cloud
            all_text = " ".join(str(text) for text in result['reviewText'] if isinstance(text, str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

            # Display Word Cloud
            st.subheader("Word Cloud of Review Text")
            fig_wordcloud, ax_wordcloud = plt.subplots(figsize=(10, 5))
            ax_wordcloud.imshow(wordcloud, interpolation='bilinear')
            ax_wordcloud.axis("off")
            st.pyplot(fig_wordcloud)


# --- Data Exploration section ---
elif selected_menu == "Data Exploration":
    st.header("Data Exploration")
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(result)
    if st.checkbox("Display data Info"):
        st.subheader("Data Information")
        st.write(result.info())
    if st.checkbox("Display Missing Values"):
        st.subheader("Missing Values")
        st.write(result.isnull().sum())
    if st.checkbox("Statistical Summary"):
        st.subheader("Statistical Description")
        st.write(result.describe())

# --- Feedback Rating Section ---
elif selected_menu == "Feedback Rating":
    st.header("Feedback Rating")
    feedback_text = st.text_area("Enter your feedback here:")

    if feedback_text:
        # Use VADER for sentiment analysis
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(feedback_text)
        compound_score = scores['compound']  # Overall sentiment score

        # Determine a rating based on the compound score
        if compound_score >= 0.5:
            rating = 5
        elif compound_score >= 0.1:
            rating = 4
        elif compound_score > -0.1:
            rating = 3
        elif compound_score <= -0.5:
            rating = 1
        elif compound_score < -0.1:
            rating = 2
        else:
            rating = 3  # Neutral rating

        st.subheader("Sentiment Analysis Result:")
        st.write(f"Estimated Rating: {rating}")
        st.write(f"Sentiment Score: {compound_score:.2f}") # Optionally display the sentiment score
        if compound_score >= 0.05:
            st.success("Overall Feedback: Positive")
        elif compound_score <= -0.05:
            st.error("Overall Feedback: Negative")
        else:
            st.info("Overall Feedback: Neutral")