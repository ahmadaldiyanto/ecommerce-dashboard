import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import datetime as dt

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df

df = load_data()

# ======================
# SIDEBAR MENU
# ======================
st.sidebar.title("📊 E-Commerce Dashboard")
menu = st.sidebar.radio(
    "Navigation",
    ["Overview",
     "Product Analysis",
     "Payment Analysis",
     "Customer Analysis",
     "RFM Analysis",
     "Customer Clustering",
     "Geolocation"]
)

# ======================
# OVERVIEW
# ======================
if menu == "Overview":
    st.title("📈 Business Overview")

    total_revenue = df["payment_value"].sum()
    total_orders = df["order_id"].nunique()
    total_customers = df["customer_unique_id"].nunique()
    avg_order_value = total_revenue / total_orders

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Total Orders", total_orders)
    col3.metric("Total Customers", total_customers)
    col4.metric("Avg Order Value", f"${avg_order_value:,.2f}")

    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    monthly = df.groupby("month").agg(
        revenue=("payment_value", "sum"),
        orders=("order_id", "nunique")
    ).reset_index()

    fig = px.line(monthly, x="month", y="revenue",
                  title="Monthly Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

# ======================
# PRODUCT ANALYSIS
# ======================
elif menu == "Product Analysis":
    st.title("🛍 Product Performance")

    category_sales = (
        df.groupby("product_category_name_english")["product_id"]
        .count()
        .reset_index(name="total_sold")
        .sort_values("total_sold", ascending=False)
        .head(10)
    )

    # Balik urutan supaya terbesar ada di atas (untuk bar horizontal)
    category_sales = category_sales.sort_values("total_sold", ascending=True)

    fig = px.bar(
        category_sales,
        x="total_sold",
        y="product_category_name_english",
        orientation="h",
        title="Top 10 Product Categories"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================
# PAYMENT ANALYSIS
# ======================
elif menu == "Payment Analysis":
    st.title("💳 Payment Method Analysis")

    payment_usage = (
        df.groupby("payment_type")["order_id"]
        .nunique()
        .reset_index()
    )

    payment_usage["percentage"] = (
        payment_usage["order_id"] /
        payment_usage["order_id"].sum()
    ) * 100

    fig = px.pie(
        payment_usage,
        names="payment_type",
        values="percentage",
        title="Payment Method Distribution (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================
# CUSTOMER ANALYSIS
# ======================
elif menu == "Customer Analysis":
    st.title("👥 Customer Insights")

    city_transactions = (
        df.groupby("customer_city")["order_id"]
        .nunique()
        .reset_index(name="total_transactions")
        .sort_values("total_transactions", ascending=False)
        .head(10)
    )

    # Balik supaya terbesar ada di atas
    city_transactions = city_transactions.sort_values("total_transactions", ascending=True)

    fig = px.bar(
        city_transactions,
        x="total_transactions",
        y="customer_city",
        orientation="h",
        title="Top 10 Cities by Transactions"
    )

    fig.update_traces(texttemplate='%{x}', textposition='outside')
    fig.update_layout(yaxis_title="", xaxis_title="Total Transactions")

    st.plotly_chart(fig, use_container_width=True)

# ======================
# RFM ANALYSIS
# ======================
elif menu == "RFM Analysis":
    st.title("📊 RFM Customer Analysis")

    snapshot_date = df["order_purchase_timestamp"].max() + dt.timedelta(days=1)

    rfm = df.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": lambda x: (snapshot_date - x.max()).days,
        "order_id": "nunique",
        "payment_value": "sum"
    }).reset_index()

    rfm.columns = ["customer_id", "Recency", "Frequency", "Monetary"]

    rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
    rfm["M_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5])

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Recency", int(rfm["Recency"].mean()))
    col2.metric("Avg Frequency", round(rfm["Frequency"].mean(),2))
    col3.metric("Avg Monetary", f"${rfm['Monetary'].mean():,.0f}")

    fig = px.histogram(rfm, x="Recency", nbins=50,
                       title="Distribution of Recency")
    st.plotly_chart(fig, use_container_width=True)

# ======================
# CUSTOMER CLUSTERING
# ======================
elif menu == "Customer Clustering":
    st.title("👥 Customer Segmentation (RFM Clustering)")

    snapshot_date = df["order_purchase_timestamp"].max() + dt.timedelta(days=1)

    rfm = df.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": lambda x: (snapshot_date - x.max()).days,
        "order_id": "nunique",
        "payment_value": "sum"
    }).reset_index()

    rfm.columns = ["customer_id", "Recency", "Frequency", "Monetary"]

    rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
    rfm["M_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5])

    def rfm_clustering(row):
        if row["R_score"] >= 4 and row["F_score"] >= 4 and row["M_score"] >= 4:
            return "High Value Customer"
        elif row["F_score"] >= 4 and row["M_score"] >= 3:
            return "Loyal Customer"
        elif row["R_score"] >= 3 and row["F_score"] >= 3:
            return "Potential Customer"
        elif row["R_score"] == 3:
            return "Need Attention"
        elif row["R_score"] <= 2 and row["F_score"] >= 3:
            return "At Risk"
        else:
            return "Lost Customer"

    rfm["Segment"] = rfm.apply(rfm_clustering, axis=1)

    segment_counts = rfm["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]

    fig = px.pie(segment_counts,
                 names="Segment",
                 values="Count",
                 title="Customer Segmentation Distribution")

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(segment_counts)

# ======================
# GEOLOCATION
# ======================
elif menu == "Geolocation":
    st.title("🌍 Customer Geolocation Map")

    if "geolocation_lat" in df.columns:
        sample = df.sample(2000)

        fig = px.scatter_mapbox(
            sample,
            lat="geolocation_lat",
            lon="geolocation_lng",
            zoom=3,
            height=600
        )

        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Geolocation data not available.")