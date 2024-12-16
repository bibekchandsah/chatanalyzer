import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import emoji

# Streamlit app configuration
st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="🔐", layout="wide")

# Function to preprocess chat data
@st.cache_data
def preprocess_chat_data(chat_data):
    data = []
    for line in chat_data.splitlines():
        if ', ' in line and ' - ' in line:  # Assuming this pattern for WhatsApp chat format
            date, rest = line.split(', ', 1)
            if ' - ' in rest:
                time, message = rest.split(' - ', 1)
                if ': ' in message:
                    user, text = message.split(': ', 1)
                    data.append([date, time, user.strip(), text.strip()])
    df = pd.DataFrame(data, columns=['Date', 'Time', 'User', 'Message'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

# Function to generate word cloud
@st.cache_data
def generate_wordcloud(text):
    return WordCloud(width=800, height=400, background_color="white").generate(text)

# Function to get messages per day
@st.cache_data
def get_messages_per_day(df):
    messages_per_day = df.groupby('Date').size().reset_index(name='Message Count')
    return messages_per_day

# File upload section
st.sidebar.header("Upload Chat Data")
uploaded_file = st.sidebar.file_uploader("Upload your WhatsApp chat file (txt format)", type="txt")

if uploaded_file:
    chat_data = uploaded_file.read().decode("utf-8")
    df = preprocess_chat_data(chat_data)

    if not df.empty:
        st.sidebar.success("File uploaded successfully!")

        # Sidebar for user selection
        unique_users = sorted(df['User'].dropna().unique())
        user_option = st.sidebar.radio("Select analysis option:", ["All Users", "Individual Users"])

        # Chat timeline
        messages_per_day = get_messages_per_day(df)
        st.header("Chat Timeline")
        fig = px.line(messages_per_day, x='Date', y='Message Count', title='Chat Timeline', labels={'Date': 'Date', 'Message Count': 'Number of Messages'})
        st.plotly_chart(fig, use_container_width=True)

        # Word Cloud section
        st.header("Word Cloud Analysis")
        if user_option == "All Users":
            all_text = " ".join(df['Message'].dropna())
            wordcloud = generate_wordcloud(all_text)
            st.image(wordcloud.to_image(), caption="Word Cloud for All Users", use_container_width=True)
        elif user_option == "Individual Users":
            selected_user = st.sidebar.selectbox("Choose a user:", unique_users)
            user_text = " ".join(df[df['User'] == selected_user]['Message'].dropna())
            wordcloud = generate_wordcloud(user_text)
            st.image(wordcloud.to_image(), caption=f"Word Cloud for {selected_user}", use_container_width=True)

        # Download button for processed data
        st.sidebar.header("Download Processed Data")
        st.sidebar.download_button(
            label="Download CSV", 
            data=df.to_csv(index=False), 
            file_name="processed_chat_data.csv",
            mime="text/csv"
        )
    else:
        st.error("The uploaded file could not be processed. Please check the format.")
else:
    st.info("Please upload a chat file to begin analysis.")

# Custom footer
st.markdown("""
    <hr>
    <footer style='text-align: center;'>
        © 2024 Chat Analyzer | Developed by Bibek Chand Sah
    </footer>
""", unsafe_allow_html=True)
