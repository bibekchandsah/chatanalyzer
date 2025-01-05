import streamlit as st
import pandas as pd
import re
from datetime import datetime, timedelta
import collections
import re
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import emoji
from collections import Counter
from textblob import TextBlob


# Function to parse WhatsApp chat with automatic time format detection
@st.cache_data
# def parse_chat(file_content):
#     lines = file_content.decode('utf-8').split("\n")
#     messages = []

#     # Regex patterns for both formats
#     pattern_24_hour = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}) - ([^:]+): (.+)'
#     pattern_am_pm = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s?[APap][Mm]) - ([^:]+): (.+)'
#     system_message_24_hour = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}) - (.+)'
#     system_message_am_pm = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s?[APap][Mm]) - (.+)'

#     for line in lines:
#         message = {}
#         # Try matching both formats for normal messages
#         match_24_hour = re.match(pattern_24_hour, line)
#         match_am_pm = re.match(pattern_am_pm, line)
#         # Try matching both formats for system messages
#         system_match_24_hour = re.match(system_message_24_hour, line)
#         system_match_am_pm = re.match(system_message_am_pm, line)

#         if match_24_hour:  # 24-hour normal messages
#             date, time, sender, content = match_24_hour.groups()
#             message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %H:%M')
#             message['sender'] = sender
#             message['message'] = content
#         elif match_am_pm:  # AM/PM normal messages
#             date, time, sender, content = match_am_pm.groups()
#             message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %I:%M %p')
#             message['sender'] = sender
#             message['message'] = content
#         elif system_match_24_hour:  # 24-hour system messages
#             date, time, content = system_match_24_hour.groups()
#             message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %H:%M')
#             message['sender'] = "System"
#             message['message'] = content
#         elif system_match_am_pm:  # AM/PM system messages
#             date, time, content = system_match_am_pm.groups()
#             message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %I:%M %p')
#             message['sender'] = "System"
#             message['message'] = content

#         if message:
#             messages.append(message)

#     return pd.DataFrame(messages)




def parse_chat(file_content):
    lines = file_content.decode('utf-8').split("\n")
    messages = []

    # Regex patterns for various formats my data not working with this
    
    # [dd/mm/yy, hh:mm:ss AM/PM] user: message
    iphone_message = r'\[(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}:\d{2}\s?[APap][Mm])\] ([^:]+): (.+)'
    # dd/mm/yy, hh:mm am/pm | AM/PM - user: message
    pattern1_dd_mm_yy_am_pm = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s?[APap][Mm]) - ([^:]+): (.+)'
    # dd/mm/yy, hh:mm am/pm | AM/PM - message
    system1_message_dd_mm_yy_am_pm = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s?[APap][Mm]) - (.+)'
    # dd/mm/yy, hh:mm - user: message
    pattern1_dd_mm_yy_24_hour = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}) - ([^:]+): (.+)'
    # dd/mm/yy, hh:mm - message
    system1_message_dd_mm_yy_24_hour = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}) - (.+)'
    # mm/dd/yy, hh:mm am/pm | AM/PM - user: message
    pattern2_mm_dd_yy_am_pm = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s?[APap][Mm]) - ([^:]+): (.+)'
    # mm/dd/yy, hh:mm am/pm | AM/PM - message
    system2_message_mm_dd_yy_am_pm = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s?[APap][Mm]) - (.+)'
    # dd/mm/yy hh.mm - user: message
    pattern_dd_mm_yy_24_hour_dot = r'(\d{1,2}/\d{1,2}/\d{2}) (\d{1,2}\.\d{2}) - ([^:]+): (.+)'

    for line in lines:
        message = {}
        line = line.replace('\u202f', ' ').replace('\xa0', ' ').replace('‎', ' ')  # Normalize spaces
        
        # Try matching both formats for normal messages
        iphoneMessage = re.match(iphone_message, line)
        match_pattern1_24_hour = re.match(pattern1_dd_mm_yy_24_hour, line)
        match_pattern1 = re.match(pattern1_dd_mm_yy_am_pm, line)
        match_pattern2 = re.match(pattern2_mm_dd_yy_am_pm, line)
        match_pattern_dot = re.match(pattern_dd_mm_yy_24_hour_dot, line)
        # Try matching both formats for system messages
        match_system1_message_24_hour = re.match(system1_message_dd_mm_yy_24_hour, line)
        match_system1_message = re.match(system1_message_dd_mm_yy_am_pm, line)
        match_system2_message = re.match(system2_message_mm_dd_yy_am_pm, line)
    
        
        if iphoneMessage:
            date, time, sender, content = iphoneMessage.groups()
            message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %I:%M:%S %p')
            message['sender'] = sender
            message['message'] = content
        elif match_pattern1_24_hour:
            date, time, sender, content = match_pattern1_24_hour.groups()
            message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %H:%M')
            message['sender'] = sender
            message['message'] = content
        # elif match_pattern1:
        #     date, time, sender, content = match_pattern1.groups()
        #     message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %I:%M %p')
        #     message['sender'] = sender
        #     message['message'] = content
        # elif match_pattern2:
        #     date, time, sender, content = match_pattern2.groups()
        #     message['date'] = datetime.strptime(date + " " + time, '%m/%d/%y %I:%M %p')
        #     message['sender'] = sender
        #     message['message'] = content
        elif match_pattern1:
            date, time, sender, content = match_pattern1.groups()
            try:
                message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %I:%M %p')
            except ValueError:
                message['date'] = datetime.strptime(date + " " + time, '%m/%d/%y %I:%M %p')
            message['sender'] = sender
            message['message'] = content
        elif match_pattern2:
            date, time, sender, content = match_pattern2.groups()
            try:
                message['date'] = datetime.strptime(date + " " + time, '%m/%d/%y %I:%M %p')
            except ValueError:
                message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %I:%M %p')
            message['sender'] = sender
            message['message'] = content
        elif match_system1_message_24_hour:  # 24-hour system messages
            date, time, content = match_system1_message_24_hour.groups()
            message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %H:%M')
            message['sender'] = "System"
            message['message'] = content
        # elif match_system1_message:
        #     date, time, content = match_system1_message.groups()
        #     message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %I:%M %p')
        #     message['sender'] = "System"
        #     message['message'] = content
        # elif match_system2_message:
        #     date, time, content = match_system2_message.groups()
        #     message['date'] = datetime.strptime(date + " " + time, '%m/%d/%y %I:%M %p')
        #     message['sender'] = "System"
        #     message['message'] = content
        elif match_system1_message:
            date, time, content = match_system1_message.groups()
            try:
                message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %I:%M %p')
            except ValueError:
                message['date'] = datetime.strptime(date + " " + time, '%m/%d/%y %I:%M %p')
            message['sender'] = "System"
            message['message'] = content
        elif match_system2_message:
            date, time, content = match_system2_message.groups()
            try:
                message['date'] = datetime.strptime(date + " " + time, '%m/%d/%y %I:%M %p')
            except ValueError:
                message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %I:%M %p')
            message['sender'] = "System"
            message['message'] = content
        elif match_pattern_dot:
            date, time, sender, content = match_pattern_dot.groups()
            message['date'] = datetime.strptime(date + " " + time, '%d/%m/%y %H.%M')
            message['sender'] = sender
            message['message'] = content

        if message:
            messages.append(message)

    return pd.DataFrame(messages)


# Function to calculate chat statistics
@st.cache_data
# def get_chat_stats(df):
#     first_message_date = df['date'].min()
#     last_message_date = df['date'].max()
#     total_days_chatted = (last_message_date - first_message_date).days
#     total_messages = len(df)
#     total_words = df['message'].str.split().str.len().sum()
#     return {
#         "first_message_date": first_message_date,
#         "last_message_date": last_message_date,
#         "total_days_chatted": total_days_chatted,
#         "total_messages": total_messages,
#         "total_words": total_words,
#     }


# def get_chat_stats(df):
#     stats = {
#         "first_message_date": df['date'].min(),
#         "last_message_date": df['date'].max(),
#         "total_days_chatted": (df['date'].max() - df['date'].min()).days + 1,
#         "total_messages": len(df),
#         "total_words": df['message'].apply(lambda x: len(x.split())).sum(),
#         "first_message": df.loc[df['date'].idxmin(), 'message'],  # First message content
#         "last_message": df.loc[df['date'].idxmax(), 'message'],   # Last message content
#         "longest_message": df.loc[df['message'].apply(len).idxmax(), 'message'],  # Longest message content
#     }
#     return stats


def get_chat_stats(df):
    # Filter out specific messages
    filtered_df = df[~df['message'].isin([
        "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them. Tap to learn more.",
        "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.",
        "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.",
        "null",
        "Your security code with Bikash Sah KIIT changed. Tap to learn more.",
        "Disappearing messages were turned off. Tap to change.",
        "<Media omitted>",
        "<Media tidak disertakan>",
        "You deleted this message"
    ])]

    stats = {
        "first_message_date": df['date'].min(),
        "last_message_date": df['date'].max(),
        "total_days_chatted": (df['date'].max() - df['date'].min()).days + 1,
        "total_messages": len(df),
        "total_words": df['message'].apply(lambda x: len(x.split())).sum(),
        "first_message": filtered_df.loc[filtered_df['date'].idxmin(), 'message'],  # First message content
        "last_message": filtered_df.loc[filtered_df['date'].idxmax(), 'message'],   # Last message content
        "longest_message": filtered_df.loc[filtered_df['message'].apply(len).idxmax(), 'message'],  # Longest message content
    }
    return stats




# Function to calculate advanced metrics
@st.cache_data
def get_advanced_metrics(df):
    # media_placeholder = "<Media omitted>"
    # deleted_message_placeholder = "This message was deleted"
    
    media_placeholder = [
        "<Media omitted>", 
        "<Media tidak disertakan>",
        "‎document omitted"
    ]
    deleted_message_placeholder = [
        "This message was deleted", 
        "You deleted this message",
        "Pesan ini dihapus"
    ]

    # Media count
    # media_count = df['message'].str.contains(media_placeholder, regex=False).sum()
    media_count = df['message'].apply(lambda x: any(placeholder in x for placeholder in media_placeholder)).sum()


    # Emoji analysis
    # Improved Emoji analysis using Unicode ranges for emojis
    emoji_pattern = r'[\U0001F600-\U0001F64F|\U0001F300-\U0001F5FF|\U0001F680-\U0001F6FF|\U0001F700-\U0001F77F|\U0001F780-\U0001F7FF|\U0001F800-\U0001F8FF|\U0001F900-\U0001F9FF|\U0001FA00-\U0001FA6F|\U0001FA70-\U0001FAFF|\U00002600-\U000026FF|\U00002700-\U000027BF|\U0001F1E0-\U0001F1FF]'

    all_emojis = ''.join(re.findall(emoji_pattern, ' '.join(df['message'].dropna())))
    emoji_counter = collections.Counter(all_emojis)
    total_emojis = sum(emoji_counter.values())
    most_used_emojis = dict(emoji_counter.most_common(5))

    # Longest message
    longest_message_length = df['message'].str.len().max()

    # Wordstock and average words per message
    all_words = ' '.join(df['message'].dropna()).split()
    unique_words = set(all_words)
    wordstock = len(unique_words)
    average_words_per_message = len(all_words) / len(df)

    # Links count
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    link_count = df['message'].str.contains(url_pattern, regex=True).sum()

    # Deleted messages count
    # deleted_messages_count = df['message'].str.contains(deleted_message_placeholder, regex=False).sum()
    deleted_messages_count = df['message'].apply(lambda x: any(placeholder in x for placeholder in deleted_message_placeholder)).sum()


    return {
        "media_count": media_count,
        "total_emojis": total_emojis,
        "most_used_emojis": most_used_emojis,
        "longest_message_length": longest_message_length,
        "wordstock": wordstock,
        "average_words_per_message": average_words_per_message,
        "link_count": link_count,
        "deleted_messages_count": deleted_messages_count,
    }

# Function to display a donut chart of the number of messages per user
@st.cache_data
def plot_message_count_donut(df):
    # Count the number of messages per user
    message_count = df['sender'].value_counts().reset_index()
    message_count.columns = ['User', 'Messages']

    # Plotting the donut chart using Plotly Express
    fig = px.pie(message_count, names='User', values='Messages', hole=0.5, title='Number of Messages per User')

    # Customize the chart layout for better presentation
    fig.update_traces(textinfo='percent+label', pull=[0.01] * len(message_count))  # Pull out slices to highlight
    fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))  # Add margins for a clean look

    return fig

# Function to generate a word cloud
@st.cache_data
def generate_wordcloud(df):
    # Define a list of words/phrases to exclude
    exclude_words = [
        'Media omitted', 
        'null', 
        'Missed voice call', 
        'Missed video call', 
        'https', 
        'security code', 
        'Media tidak', 
        'tidak disertakan', 
        'disertakan Media',
        'answer',
        'Voice',
        'Voice call',
        'call',
        'sec',
    ]
    
    # Combine all messages into a single string
    text = " ".join(df['message'].dropna())

    # Remove unwanted words
    for word in exclude_words:
        text = text.replace(word, "")
        
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Display the word cloud
    st.image(wordcloud.to_array(), caption=f"Word Cloud for {selected_user}", use_container_width=True)
    
    # Plot the word cloud using matplotlib
    # plt.figure(figsize=(10, 5))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')  # No axes for a cleaner look
    # plt.show()
    
# Regex pattern for detecting emojis
EMOJI_PATTERN = re.compile("[\U0001F600-\U0001F64F"
                           "\U0001F300-\U0001F5FF"
                           "\U0001F680-\U0001F6FF"
                           "\U0001F700-\U0001F77F"
                           "\U0001F780-\U0001F7FF"
                           "\U0001F800-\U0001F8FF"
                           "\U0001F900-\U0001F9FF"
                           "\U0001FA00-\U0001FA6F"
                           "\U0001FA70-\U0001FAFF"
                           "\U00002702-\U000027B0"
                           "\U000024C2-\U0001F251]+", re.UNICODE)

# Function to perform emoji analysis and display the top 10 emojis as a pie chart and table
@st.cache_data
def extract_emojis(text):
    return [char for char in text if char in emoji.EMOJI_DATA]

@st.cache_data
def get_emoji_stats(filtered_df, top_n=10):
    all_emojis = []
    for message in filtered_df['message']:
        all_emojis.extend(extract_emojis(message))
    
    emoji_counts = Counter(all_emojis)
    # Return the top N emojis as a dictionary
    return dict(emoji_counts.most_common(top_n))

@st.cache_data
def plot_emoji_pie_chart(emoji_counts):
    emoji_df = pd.DataFrame(emoji_counts.items(), columns=['Emoji', 'Count'])
    fig = px.pie(
        emoji_df,
        names='Emoji',
        values='Count',
        title='Top 10 Emojis Used in Chat',
        hole=0.01,
    )
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))
    # fig.update_traces(insidetextorientation='radial')
    return fig

@st.cache_data
def get_emoji_table(filtered_df):
    all_emojis = []
    for message in filtered_df['message']:
        all_emojis.extend(extract_emojis(message))
    
    emoji_counts = Counter(all_emojis)
    # Convert to a pandas DataFrame
    emoji_table = pd.DataFrame(emoji_counts.items(), columns=["Emoji", "Count"])
    # Sort by count in descending order
    emoji_table = emoji_table.sort_values(by="Count", ascending=False).reset_index(drop=True)
    return emoji_table


@st.cache_data
def display_row_wise_emoji_table(emoji_table):
    if not emoji_table.empty:
        # Convert the table into row-wise format (without serial numbers)
        row_wise_table = pd.DataFrame([
            emoji_table['Emoji'].values,
            emoji_table['Count'].values
        ], index=['Emoji', 'Count'])
        
        # Display the row-wise table
        st.table(row_wise_table)
    else:
        st.write("No emojis found.")




# Function to plot hourly message activity
@st.cache_data
def hourly_message_activity(df):
    # Ensure 'date' is in datetime format (if it's not already)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract the hour from the 'date' column
    df['hour'] = df['date'].dt.hour
    
    # Count the number of messages sent in each hour
    hourly_activity = df.groupby('hour').size().reset_index(name='message_count')
    
    # Plot the hourly message activity as a bar chart using Plotly
    fig = px.bar(hourly_activity, x='hour', y='message_count', 
                 labels={'hour': 'At Hour', 'message_count': 'Number of Messages'},
                 title='Hourly Message Activity', color_discrete_sequence=["#AB63FA"])
    
    # Update the layout for better aesthetics
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1, title=f'Hour Activity of {selected_user}'), showlegend=False)
    
    # Display the graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)
       
    
# Function to plot monthly message activity with sorted x-axis
@st.cache_data
def monthly_message_activity_with_names(df):
    # Ensure 'date' is in datetime format (if it's not already)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract the year and month
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month  # Numeric month for sorting
    df['month_name'] = df['date'].dt.strftime('%B')  # Full month name (e.g., January, February)
    
    # Count the number of messages per year and month
    monthly_activity = df.groupby(['year', 'month', 'month_name']).size().reset_index(name='message_count')
    
    # Sort the DataFrame by year and month
    monthly_activity = monthly_activity.sort_values(by=['year', 'month'])
    
    # Create a combined column for "Year - Month" (e.g., "2024 - January")
    monthly_activity['year_month'] = monthly_activity['year'].astype(str) + " - " + monthly_activity['month_name']
    
    # Plot the sorted monthly message activity as a bar chart using Plotly
    fig = px.bar(monthly_activity, x='year_month', y='message_count',
                 labels={'year_month': 'Month', 'message_count': 'Number of Messages'},
                 title='Monthly Message Activity', color_discrete_sequence=["#EF553B"])
    
    # Update the layout for better aesthetics
    fig.update_layout(xaxis=dict(tickmode='linear', title=f'Monthly Activity of {selected_user}'), showlegend=False)
    
    # Display the graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Function to plot weekday message activity
@st.cache_data
def weekday_message_activity(df):
    # Ensure 'date' is in datetime format (if it's not already)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract the weekday name (e.g., Monday, Tuesday) and store it in a new column
    df['weekday'] = df['date'].dt.strftime('%A')
    
    # Count the number of messages per weekday
    weekday_activity = df['weekday'].value_counts().reset_index()
    weekday_activity.columns = ['weekday', 'message_count']
    
    # Order the weekdays (Monday to Sunday)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_activity['weekday'] = pd.Categorical(weekday_activity['weekday'], categories=weekday_order, ordered=True)
    weekday_activity = weekday_activity.sort_values('weekday')  # Sort by weekday order
    
    # Plot the weekday message activity as a bar chart using Plotly
    fig = px.bar(weekday_activity, x='weekday', y='message_count',
                 labels={'weekday': 'Weekday', 'message_count': 'Number of Messages'},
                 title='Weekday Message Activity', color_discrete_sequence=["#00CC96"])
    
    # Update the layout for better aesthetics
    fig.update_layout(xaxis=dict(tickmode='linear', title=f'Weedkay Activity of {selected_user}'), showlegend=False)
    
    # Display the graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    

# Function to plot chat timeline activity
@st.cache_data
def chat_timeline(df):
    # Ensure 'date' is in datetime format (if it's not already)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Group by date and count the number of messages
    timeline_activity = df.groupby(df['date'].dt.date).size().reset_index(name='message_count')
    
    # Rename columns for clarity
    timeline_activity.columns = ['date', 'message_count']
    
    # Sort by date
    timeline_activity = timeline_activity.sort_values('date')
    
    # Plot the timeline graph using Plotly
    fig = px.line(timeline_activity, x='date', y='message_count',
                  labels={'date': 'Date', 'message_count': 'Number of Messages'},
                  title='Chat Timeline Activity', color_discrete_sequence=["#636efa"])
    
    # Update layout for better aesthetics
    fig.update_layout(
        xaxis=dict(tickformat='%d-%b-%Y', title=f'Timeline Activity of {selected_user}'),
        yaxis=dict(title='Number of Messages'),
        showlegend=False
    )
    
    # Display the graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Function to perform sentiment analysis
# @st.cache_data
# def sentiment_analysis(df):
#     # Create a new column 'sentiment' using TextBlob
#     def get_sentiment(message):
#         analysis = TextBlob(message)
#         if analysis.sentiment.polarity > 0:
#             return 'Positive'
#         elif analysis.sentiment.polarity < 0:
#             return 'Negative'
#         else:
#             return 'Neutral'

#     # Apply sentiment analysis to the 'message' column
#     df['sentiment'] = df['message'].apply(lambda msg: get_sentiment(msg) if pd.notnull(msg) else 'Neutral')

#     # Count sentiment types
#     sentiment_counts = df['sentiment'].value_counts().reset_index()
#     sentiment_counts.columns = ['Sentiment', 'Count']

#     # Plot the sentiment distribution
#     fig = px.pie(sentiment_counts, names='Sentiment', values='Count',
#                  title='Sentiment Analysis',
#                  color='Sentiment',
#                  color_discrete_map={'Positive': '#00cc96', 'Neutral': 'royalblue', 'Negative': 'red'})
    
#     # Display the chart in Streamlit
#     st.plotly_chart(fig, use_container_width=True)

#     # Show sentiment counts in table format
#     # st.write("### Sentiment Breakdown")
#     # st.table(sentiment_counts)


# # Function to show top positive and negative messages
# @st.cache_data
# def top_positive_negative_messages(df, n=5):
#     # Add a polarity column using TextBlob
#     def get_polarity(message):
#         return TextBlob(message).sentiment.polarity if pd.notnull(message) else 0

#     # Create the polarity column
#     df['polarity'] = df['message'].apply(get_polarity)

#     # Round the polarity column to 2 decimal places
#     df['polarity'] = df['polarity'].round(2)

#     # Extract top positive and negative messages
#     top_positive = df.nlargest(n, 'polarity')[['date', 'sender', 'message', 'polarity']]
#     top_negative = df.nsmallest(n, 'polarity')[['date', 'sender', 'message', 'polarity']]

#     # Format polarity for display without affecting the numeric column
#     top_positive['polarity'] = top_positive['polarity'].apply(lambda x: f"{x:.2f}")
#     top_negative['polarity'] = top_negative['polarity'].apply(lambda x: f"{x:.2f}")

#     # Display in Streamlit
#     st.write("### Top Positive Messages")
#     st.table(top_positive)

#     st.write("### Top Negative Messages")
#     st.table(top_negative)



# Messages to exclude
exclude_messages = [
    "Disappearing messages were turned off. Tap to change.",
    "Voice call, Answered on other device",
]

# Starting phrases to exclude
exclude_startswith = (
    "Your security code with",
    "Voice call",
)

def should_exclude(message):
    return any(message.startswith(prefix) for prefix in exclude_startswith) or message in exclude_messages

# Function to perform sentiment analysis
@st.cache_data
def sentiment_analysis(df):
    # Create a new column 'sentiment' using TextBlob
    def get_sentiment(message):
        analysis = TextBlob(message)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    # Apply sentiment analysis to the 'message' column, excluding specific messages
    df['sentiment'] = df['message'].apply(lambda msg: get_sentiment(msg) if pd.notnull(msg) and not should_exclude(msg) else 'Neutral')

    # Count sentiment types
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    # Plot the sentiment distribution
    fig = px.pie(sentiment_counts, names='Sentiment', values='Count',
                 title='Sentiment Analysis',
                 color='Sentiment',
                 color_discrete_map={'Positive': '#00cc96', 'Neutral': 'royalblue', 'Negative': 'red'})
    
    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Show sentiment counts in table format
    # st.write("### Sentiment Breakdown")
    # st.table(sentiment_counts)


# Function to show top positive and negative messages
@st.cache_data
def top_positive_negative_messages(df, n=5):
    # Add a polarity column using TextBlob
    def get_polarity(message):
        return TextBlob(message).sentiment.polarity if pd.notnull(message) and not should_exclude(message) else 0

    # Create the polarity column
    df['polarity'] = df['message'].apply(get_polarity)

    # Round the polarity column to 2 decimal places
    df['polarity'] = df['polarity'].round(2)

    # Extract top positive and negative messages
    top_positive = df.nlargest(n, 'polarity')[['date', 'sender', 'message', 'polarity']]
    top_negative = df.nsmallest(n, 'polarity')[['date', 'sender', 'message', 'polarity']]

    # Format polarity for display without affecting the numeric column
    top_positive['polarity'] = top_positive['polarity'].apply(lambda x: f"{x:.2f}")
    top_negative['polarity'] = top_negative['polarity'].apply(lambda x: f"{x:.2f}")

    # Display in Streamlit
    st.write("### Top Positive Messages")
    st.table(top_positive)

    st.write("### Top Negative Messages")
    st.table(top_negative)




# Function to filter messages by date or time range
# @st.cache_data
def filter_messages_by_date_or_time(df):
    # Convert 'date' column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    st.write("### Filter Messages by Date and/or Time Range")

    # Date Range Picker
    date_filter_enabled = st.checkbox("Filter by Date Range")
    if date_filter_enabled:
        start_date = st.date_input("Start Date", value=df['date'].min().date(), key="start_date")
        end_date = st.date_input("End Date", value=df['date'].max().date(), key="end_date")
        if start_date > end_date:
            st.error("End Date must be after Start Date.")
            return None

    # Hour Range Slider
    time_filter_enabled = st.checkbox("Filter by Time Range")
    if time_filter_enabled:
        hour_range = st.slider(
            "Select Hour Range",
            min_value=0,
            max_value=23,
            value=(0, 23),  # Default range
            step=1,
            key="hour_range"
        )
        start_hour, end_hour = hour_range

    # Filtering Logic
    filtered_df = df.copy()

    if date_filter_enabled:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & (filtered_df['date'].dt.date <= end_date)
        ]

    if time_filter_enabled:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.hour >= start_hour) & (filtered_df['date'].dt.hour <= end_hour)
        ]

    # Display the filtered messages
    if date_filter_enabled or time_filter_enabled:
        st.write("### Filtered Messages")
        st.dataframe(filtered_df[['date', 'sender', 'message']])
    else:
        st.write("Please select at least one filter (date or time).")

    return filtered_df


# @st.cache_data
def filter_messages_by_additional_options(df):
    # st.write("### Filter Messages by Additional Options")

    # Multi-Select Dropdown for additional filter options with no default selection
    filter_options = [
        "Media Shared", "Emoji", "Link", 
        "Message Deleted", "Video Call", "Voice Call"
    ]
    selected_filters = st.multiselect(
        "Select Filters", filter_options, default=[]  # No default selection
    )

    filtered_df = df.copy()

    # Apply filters based on selected options
    if selected_filters:
        # Initialize an empty condition (True for all rows)
        condition = pd.Series(False, index=filtered_df.index)  # Matches the current index

        # Apply "Media Shared" filter
        if "Media Shared" in selected_filters:
            condition |= filtered_df['message'].str.contains('media', case=False, na=False)

        # Apply "Emoji" filter
        if "Emoji" in selected_filters:
            condition |= filtered_df['message'].str.contains(r'[\U00010000-\U0010ffff]', na=False)

        # Apply "Link" filter
        if "Link" in selected_filters:
            condition |= filtered_df['message'].str.contains(r'http[s]?://', na=False)

        # Apply "Message Deleted" filter
        if "Message Deleted" in selected_filters:
            condition |= filtered_df['message'].str.contains('This message was deleted', na=False)

        # Apply "Video Call" filter
        if "Video Call" in selected_filters:
            condition |= filtered_df['message'].str.contains('Missed video call', case=False, na=False)

        # Apply "Voice Call" filter
        if "Voice Call" in selected_filters:
            condition |= filtered_df['message'].str.contains('Missed voice call', case=False, na=False)

        # Apply the condition to filter the messages
        filtered_df = filtered_df[condition]

        # Display the filtered messages
        st.write("### Filtered Messages Based on Selected Options")
        st.dataframe(filtered_df[['date', 'sender', 'message']])
    else:
        st.write("Please select at least one filter option.")

    return filtered_df








# Streamlit app layout
# st.title("WhatsApp Chat Analyzer")
# Streamlit app configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer", 
    # page_icon="🗨️", 
    page_icon="🔐", 
    # page_icon="https://cdn-icons-png.flaticon.com/512/1674/1674691.png", 
    # page_icon="https://cdn3d.iconscout.com/3d/premium/thumb/analysis-message-5637729-4699021.png", 
    # page_icon="https://cdn-icons-png.flaticon.com/512/1341/1341841.png", 
    layout="wide")

# Sidebar layout
st.sidebar.header("Upload File and Select User")
uploaded_file = st.sidebar.file_uploader("Upload WhatsApp chat file (.txt)", type="txt")

# Integrating with Streamlit
if uploaded_file:
    # file_content = uploaded_file.read().decode("utf-8")
    file_content = uploaded_file.read()
    df = parse_chat(file_content)
    
    if df.empty:
        st.error("The uploaded file does not match the expected format. Please check your file and try again.")
    
    if not df.empty:
        st.sidebar.success("File uploaded successfully!")

        # Get unique usernames
        unique_users = sorted(df['sender'].unique())
        unique_users.insert(0, "All Users")

        # Dropdown for user selection in the sidebar
        selected_user = st.sidebar.selectbox("Select a User", unique_users)

        # Filter the data based on the selected user
        if selected_user != "All Users":
            filtered_df = df[df['sender'] == selected_user]
        else:
            filtered_df = df
        
        # Download button for processed data
        st.sidebar.header("Download Processed Data")
        st.sidebar.download_button(
            label="Download CSV", 
            data=df.to_csv(index=False), 
            file_name="processed_chat_data.csv",
            mime="text/csv"
        )

        # Display chat statistics
        stats = get_chat_stats(filtered_df)
        advanced_metrics = get_advanced_metrics(filtered_df)

        # st.subheader(f"Chat Statistics for {selected_user}")
        # st.write(f"**First Message:** {stats['first_message_date'].strftime('%A, %B %d, %Y')}")
        # st.write(f"**Last Message:** {stats['last_message_date'].strftime('%A, %B %d, %Y')}")
        # st.write(f"**No. of days chatted:** {stats['total_days_chatted']} days "
        #          f"({stats['total_days_chatted'] // 365} years, "
        #          f"{(stats['total_days_chatted'] % 365) // 30} months, "
        #          f"{(stats['total_days_chatted'] % 365) % 30} days)")
        # st.write(f"**No. of messages exchanged:** {stats['total_messages']} messages")
        # st.write(f"**Total words:** {stats['total_words']} words")

        # st.subheader(f"Advanced Chat Metrics for {selected_user}")
        # st.write(f"**Number of emojis:** {advanced_metrics['total_emojis']}")
        # st.write(f"**Most used emojis:** {advanced_metrics['most_used_emojis']}")
        # st.write(f"**Longest message:** {advanced_metrics['longest_message_length']} characters")
        # st.write(f"**Wordstock (unique words used):** {advanced_metrics['wordstock']}")
        # st.write(f"**Average words per message:** {advanced_metrics['average_words_per_message']:.2f}")
        # st.write(f"**Number of media shared:** {advanced_metrics['media_count']}")
        # st.write(f"**Number of links:** {advanced_metrics['link_count']}")
        # st.write(f"**Number of messages deleted:** {advanced_metrics['deleted_messages_count']}")
        
        
        
        # Detect theme
        # if "theme" in st.session_state:
        #     current_theme = st.session_state["theme"]["base"]
        # else:
        #     current_theme = "dark"  # Default theme
        
        # Detect theme
        current_theme = st.session_state.get("theme", {}).get("base", "dark")
            
        # Define background color based on theme
        # body_bg_color = "#ffffff" if current_theme == "light" else "#121212"

        # Inject CSS to change body background color
        # st.markdown(f"""
        #     <style>
        #         body {{
        #             background-color: {body_bg_color};
        #         }}
        #         .stApp {{
        #             background-color: {body_bg_color};
        #         }}
        #     </style>
        # """, unsafe_allow_html=True)
        
        # Define styles based on theme
        bg_color = "#f4f4f4" if current_theme == "light" else "#2c2c2c"
        text_color = "#000000" if current_theme == "light" else "#ffffff"

        # Chat Statistics Section
        st.subheader(f"Chat Statistics for {selected_user}")
        # Divide into two columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>First Message:</strong><br>{stats['first_message_date'].strftime('%A, %B %d, %Y')}
            </div>
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Last Message:</strong><br>{stats['last_message_date'].strftime('%A, %B %d, %Y')}
            </div>
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Days Chatted:</strong><br>{stats['total_days_chatted']} days<br>
                ({stats['total_days_chatted'] // 365} years, {(stats['total_days_chatted'] % 365) // 30} months, {(stats['total_days_chatted'] % 365) % 30} days)
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Messages Exchanged:</strong><br>{stats['total_messages']} messages
            </div>
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Total Words:</strong><br>{stats['total_words']} words
            </div>
            """, unsafe_allow_html=True)

        # Advanced Chat Metrics Section
        st.subheader(f"Advanced Chat Metrics for {selected_user}")
        # Divide into two columns for advanced metrics
        col3, col4 = st.columns(2)

        with col3:
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Total Emojis:</strong><br>{advanced_metrics['total_emojis']}
            </div>
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Most Used Emojis:</strong><br>{advanced_metrics['most_used_emojis']}
            </div>
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Longest Message:</strong><br>{advanced_metrics['longest_message_length']} characters
            </div>
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Wordstock (Unique Words):</strong><br>{advanced_metrics['wordstock']}
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Average Words per Message:</strong><br>{advanced_metrics['average_words_per_message']:.2f}
            </div>
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Media Shared:</strong><br>{advanced_metrics['media_count']}
            </div>
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Links:</strong><br>{advanced_metrics['link_count']}
            </div>
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Messages Deleted:</strong><br>{advanced_metrics['deleted_messages_count']}
            </div>
            """, unsafe_allow_html=True)
        
        
        st.subheader(f"Special Chat Statistics for {selected_user}")
        # Divide into two columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>First Message:</strong><br>
                <em>{stats['first_message']}</em><br>
                <small>{stats['first_message_date'].strftime('%A, %B %d, %Y')}</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
                <strong>Last Message:</strong><br>
                <em>{stats['last_message']}</em><br>
                <small>{stats['last_message_date'].strftime('%A, %B %d, %Y')}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding: 10px; border-radius: 10px; background-color: {bg_color}; color: {text_color}; margin-bottom: 10px;">
            <strong>Longest Message:</strong><br>
            <em>{stats['longest_message']}</em><br>
            <small>{len(stats['longest_message'])} characters</small>
        </div> 
        """, unsafe_allow_html=True)
        
        
        

        # Plotting the donut chart for message count
        st.subheader(f"Message Count for {selected_user}")
        fig = plot_message_count_donut(filtered_df)
        st.plotly_chart(fig, key="message_count_donut")

        # Basic analysis (Donut chart of user message count)
        # st.subheader("Basic Analysis")
        # if selected_user == "All Users":
        #     user_message_count = filtered_df['sender'].value_counts()
        #     fig = plot_message_count_donut(filtered_df)
        #     st.plotly_chart(fig, key="user_message_count_donut")
        # else:
        #     st.write("No additional visualization available for a single user.")

        # Display emoji Analysis
        st.subheader(f"Emoji Analysis for {selected_user}")
        # Get emoji stats in pie chart for the selected user
        emoji_counts = get_emoji_stats(filtered_df, top_n=10)
        if emoji_counts:
            # Generate pie chart
            fig = plot_emoji_pie_chart(emoji_counts)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"No emojis found in the messages for {selected_user}.")
            
        # Generate the emoji table
        # emoji_table = get_emoji_table(filtered_df)
        # if not emoji_table.empty:
        #     st.dataframe(emoji_table, use_container_width=True)
        # else:
        #     st.write(f"No emojis found in the messages for {selected_user}.")
        
        # Generate the emoji table in row-wise
        st.subheader(f"Emoji Usage Table for {selected_user}")
        # Generate the emoji table
        emoji_table = get_emoji_table(filtered_df)  # Assuming this gets 'Emoji' and 'Count' columns
        # Display in row-wise format
        display_row_wise_emoji_table(emoji_table)



        # Generate and display the word cloud
        st.subheader(f"Word Cloud for {selected_user}")
        generate_wordcloud(filtered_df)

        # Display chat timeline 
        st.subheader(f"Chat timeline message for {selected_user}")
        chat_timeline(filtered_df)

        # Display filtered according to monthly
        st.subheader(f"Monthly message for {selected_user}")
        monthly_message_activity_with_names(filtered_df)

        # Display filtered according to weeekly
        st.subheader(f"Weekly message for {selected_user}")
        weekday_message_activity(filtered_df)

        # Display filtered according to hourly
        st.subheader(f"Hourly message for {selected_user}")
        hourly_message_activity(filtered_df)

        # Display Sentiment Analysis
        st.subheader(f"Sentiment Alalysis for {selected_user}")
        sentiment_analysis(filtered_df)

        # Display Top Sentiment Analysis
        st.subheader(f"Top Sentiment Alalysis for {selected_user}")
        top_positive_negative_messages(filtered_df)

        # Display Filtered chat according to date or time
        st.subheader(f"Filtered Message for {selected_user}")
        filtered_by_date_or_time_df = filter_messages_by_date_or_time(filtered_df)

        # Display Filtered chat additional options
        st.subheader(f"Filtered Message additional for {selected_user}")
        filtered_by_additional_options_df = filter_messages_by_additional_options(filtered_df)  

        # Display all chat data
        st.subheader(f"Messages for {selected_user}")
        st.write(filtered_df)
        
    else:
        st.error("The uploaded file could not be processed. Please check the format.")
    
    
else:
    st.info("Please upload a chat file to begin analysis.")

# Custom footer
import datetime
current_year = datetime.datetime.now().year
st.markdown(f"""
    <hr>
    <footer style='text-align: center;'>
        © {current_year} Chat Analyzer | Developed by <a href='https://bibekchandsah.com.np' target='_blank' style='text-decoration: none; color: inherit;'>Bibek Chand Sah</a>
    </footer>
""", unsafe_allow_html=True)
