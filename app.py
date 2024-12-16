import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import emoji
import re

# Define preprocess_chat_data function here (as shown above)
def preprocess_chat_data(chat_data):
    """
    Preprocess raw WhatsApp chat data into a structured DataFrame.
    """
    # Regex to parse chat lines
    chat_pattern = r'(\d{2}/\d{2}/\d{2,4}), (\d{1,2}:\d{2}\s[apm]{2}) - ([^:]+?): (.+)'
    
    messages = []
    for line in chat_data.split('\n'):
        match = re.match(chat_pattern, line)
        if match:
            date, time, user, message = match.groups()
            messages.append({'Date': date, 'Time': time, 'User': user.strip(), 'Message': message.strip()})
    
    # Convert to DataFrame
    df = pd.DataFrame(messages)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
    return df

st.title("WhatsApp Chat Analyzer")

# Upload chat file
uploaded_file = st.file_uploader("Upload your chat file", type="txt")
if uploaded_file:
    # Parse chat data
    chat_data = uploaded_file.read().decode('utf-8')
    df = preprocess_chat_data(chat_data)

    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Select an Analysis Option", ["Timeline", "Word Cloud", "Word Frequency", "Emoji Frequency"])

    if option == "Timeline":
        messages_per_day = df.groupby('Date').size().reset_index(name='Message Count')
        fig = px.line(messages_per_day, x='Date', y='Message Count', title='Chat Timeline',
                      labels={'Date': 'Date', 'Message Count': 'Number of Messages'})
        st.plotly_chart(fig)

    elif option == "Word Cloud":
        user = st.sidebar.selectbox("Select User", df['User'].unique())
        user_messages = " ".join(df[df['User'] == user]['Message'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_messages)
        st.image(wordcloud.to_array(), caption=f"Word Cloud for {user}")

    elif option == "Word Frequency":
        user = st.sidebar.selectbox("Select User", df['User'].unique())
        user_messages = " ".join(df[df['User'] == user]['Message'])
        all_words = user_messages.split()
        word_counts = Counter(all_words).most_common(10)
        words, counts = zip(*word_counts)

        st.bar_chart(pd.DataFrame({'Words': words, 'Counts': counts}))

    elif option == "Emoji Frequency":
        user = st.sidebar.selectbox("Select User", df['User'].unique())
        user_messages = " ".join(df[df['User'] == user]['Message'])
        all_emojis = [char for message in user_messages for char in message if char in emoji.EMOJI_DATA]
        emoji_counts = Counter(all_emojis).most_common(10)
        emojis, counts = zip(*emoji_counts)

        st.write("Top Emojis")
        st.write(pd.DataFrame({'Emoji': emojis, 'Count': counts}))

        fig = px.pie(names=emojis, values=counts, title='Emoji Frequency')
        st.plotly_chart(fig)
