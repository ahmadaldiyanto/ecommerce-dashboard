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
# SIDEBAR - GLOBAL FILTER (INTERACTIVE FEATURE)
# ======================
st.sidebar.title("📊 E-Commerce Dashboard")

st.sidebar.markdown("## 🔎 Filter Data")

min_date = df["order_purchase_timestamp"].min()
max_date = df["order_purchase_timestamp"].max()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Apply date filter
df = df[
    (df["order_purchase_timestamp"] >= pd.to_datetime(start_date)) &
    (df["order_purchase_timestamp"] <= pd.to_datetime(end_date))
]

# Optional category filter
if "product_category_name_english" in df.columns:
    category_list = df["product_category_name_english"].dropna().unique()
    selected_category = st.sidebar.selectbox(
        "Filter by Product Category (Optional)",
        ["All"] + sorted(category_list.tolist())
    )

    if selected_category != "All":
        df = df[df["product_category_name_english"] == selected_category]

# ======================
# NAVIGATION
# ======================
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
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

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

    fig = px.line(
        monthly,
        x="month",
        y=["revenue", "orders"],
        markers=True,
        title="Monthly Performance Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================
# PRODUCT ANALYSIS
# ======================
elif menu == "Product Analysis":
    st.title("🛍 Product Performance")

    category_sales = (
        df.groupby("product_category_name_english")["order_item_id"]
        .count()
        .reset_index(name="total_sold")
        .sort_values("total_sold", ascending=False)
        .head(10)
    )

    fig = px.bar(
        category_sales,
        x="total_sold",
        y="product_category_name_english",
        orientation="h",
        title="Top 10 Product Categories"
    )

    fig.update_layout(yaxis_title="", xaxis_title="Total Items Sold")

    st.plotly_chart(fig, use_container_width=True)

# ======================
# PAYMENT ANALYSIS
# ======================
elif menu == "Payment Analysis":
    st.title("💳 Payment Method Analysis")

    payment_usage = (
        df.groupby("payment_type")["order_id"]
        .nunique()
        .reset_index(name="total_transactions")
    )

    payment_usage["percentage"] = (
        payment_usage["total_transactions"] /
        payment_usage["total_transactions"].sum()
    ) * 100

    fig = px.pie(
        payment_usage,
        names="payment_type",
        values="percentage",
        hole=0.4,
        title="Payment Method Distribution (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================
# CUSTOMER ANALYSIS
# ======================
elif menu == "Customer Analysis":
    st.title("👥 Customer Insights")

    city_customers = (
        df.groupby("customer_city")["customer_unique_id"]
        .nunique()
        .reset_index(name="total_customers")
        .sort_values("total_customers", ascending=False)
        .head(10)
    )

    fig = px.bar(
        city_customers,
        x="total_customers",
        y="customer_city",
        orientation="h",
        title="Top 10 Cities by Unique Customers"
    )

    fig.update_layout(yaxis_title="", xaxis_title="Total Customers")

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

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Recency (Days)", int(rfm["Recency"].mean()))
    col2.metric("Avg Frequency", round(rfm["Frequency"].mean(), 2))
    col3.metric("Avg Monetary", f"${rfm['Monetary'].mean():,.0f}")

    fig = px.histogram(
        rfm,
        x="Recency",
        nbins=50,
        title="Distribution of Recency"
    )

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

    fig = px.pie(
        segment_counts,
        names="Segment",
        values="Count",
        hole=0.4,
        title="Customer Segmentation Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(segment_counts)

# ======================
# GEOLOCATION
# ======================
elif menu == "Geolocation":
    st.title("🌍 Customer Geolocation Map")

    if "geolocation_lat" in df.columns:
        sample = df.sample(min(2000, len(df)))

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