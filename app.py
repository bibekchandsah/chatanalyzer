# individual surface data

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import emoji
import re
from dateutil.relativedelta import relativedelta
from collections import Counter
from textblob import TextBlob


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

# Function to perform sentiment analysis
@st.cache_data
def perform_sentiment_analysis(df):
    """
    Perform sentiment analysis on messages in the DataFrame.
    Add sentiment scores and polarity labels (Positive, Neutral, Negative).
    """
    def classify_sentiment(polarity):
        if polarity > 0:
            return 'Positive'
        elif polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    df['Polarity'] = df['Message'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)
    df['Sentiment'] = df['Polarity'].apply(classify_sentiment)
    return df

# Emoji Ananlysis
def extract_emojis(text):
    """Extract all emojis from a text."""
    return [char for char in text if char in emoji.EMOJI_DATA]

def calculate_emoji_usage(df):
    """Calculate emoji usage for all users and each unique user."""
    df['Emojis'] = df['Message'].apply(lambda msg: extract_emojis(msg) if isinstance(msg, str) else [])
    
    all_emojis = [emoji for emojis in df['Emojis'] for emoji in emojis]
    emoji_count_all = Counter(all_emojis).most_common(10)  # Top 10 emojis across all users

    user_emoji_data = {}
    for user in df['User'].unique():
        user_emojis = [emoji for emojis in df[df['User'] == user]['Emojis'] for emoji in emojis]
        user_emoji_data[user] = Counter(user_emojis).most_common(10)  # Top 10 emojis for the user

    return emoji_count_all, user_emoji_data


# Function to generate word cloud
@st.cache_data
def generate_wordcloud(text):
    return WordCloud(width=800, height=400, background_color="white").generate(text)

# Function to get messages per day
@st.cache_data
def get_messages_per_day(df):
    messages_per_day = df.groupby('Date').size().reset_index(name='Message Count')
    return messages_per_day

# Function to calculate individual user stats
def calculate_user_stats(df, user):
    user_df = df[df['User'] == user]
    total_words = user_df['Message'].str.split().str.len().sum()
    # Extract emojis from messages and count their occurrences
    all_emojis = "".join([char for message in user_df['Message'] for char in message if char in emoji.EMOJI_DATA])
    most_used_emojis = pd.Series(list(all_emojis)).value_counts().head(5).to_dict()
    longest_message = user_df['Message'].str.len().max()
    wordstock = len(set(" ".join(user_df['Message']).split()))
    avg_words_per_message = total_words / len(user_df) if len(user_df) > 0 else 0
    no_of_media = user_df['Message'].str.contains('<Media omitted>').sum()
    no_of_emojis = user_df['Message'].apply(lambda x: len(emoji.emoji_list(x))).sum()
    no_of_links = user_df['Message'].str.contains('http').sum()
    no_of_deleted_messages = user_df['Message'].str.contains('This message was deleted').sum()

    return {
        "Total words": total_words,
        "Most used emojis": most_used_emojis,
        "Longest message": longest_message,
        "Wordstock (unique words used)": wordstock,
        "Average words per message": avg_words_per_message,
        "Number of media shared": no_of_media,
        "Number of emojis": no_of_emojis,
        "Number of links": no_of_links,
        "Number of messages deleted": no_of_deleted_messages
    }

# Function to get the first and last message dates
def get_first_last_message_dates(df):
    first_message_date = df['Date'].min().strftime("%A, %B %d, %Y")
    last_message_date = df['Date'].max().strftime("%A, %B %d, %Y")
    return first_message_date, last_message_date

def get_chat_summary(df):
    total_days = (df['Date'].max() - df['Date'].min()).days + 1
    total_messages = len(df)
    return total_days, total_messages

# Function to calculate the duration in years, months, and days
def format_duration_in_years_months_days(start_date, end_date):
    duration = relativedelta(end_date, start_date)
    return f"{duration.years} year{'s' if duration.years != 1 else ''}, " \
           f"{duration.months} month{'s' if duration.months != 1 else ''}, " \
           f"and {duration.days} day{'s' if duration.days != 1 else ''}"


# File upload section
st.sidebar.header("Upload Chat Data")
uploaded_file = st.sidebar.file_uploader("Upload your WhatsApp chat file (txt format)", type="txt")

if uploaded_file:
    chat_data = uploaded_file.read().decode("utf-8")
    df = preprocess_chat_data(chat_data)
    
    if df.empty:
        st.error("The uploaded file does not match the expected format. Please check your file and try again.")


    if not df.empty:
        st.sidebar.success("File uploaded successfully!")
        
        # Chat Summary
        # Chat Duration and Message Count
        st.header("Chat Summary")
        first_date = df['Date'].min()
        last_date = df['Date'].max()
        total_days = (last_date - first_date).days + 1
        duration_formatted = format_duration_in_years_months_days(first_date, last_date)
        total_messages = len(df)
        
        # First and Last Message Dates
        first_date, last_date = get_first_last_message_dates(df)
        st.write(f"**First Message:** {first_date}")
        st.write(f"**Last Message:** {last_date}")
        # no. of days & message exchanged
        total_days, total_messages = get_chat_summary(df)
        # st.write(f"**No. of days chatted:** {total_days} days")
        # st.write(f"**No. of messages exchanged:** {total_messages} messages")
        st.write(f"**No. of days chatted:** {total_days} days ({duration_formatted})")
        st.write(f"**No. of messages exchanged:** {total_messages} messages")

        # Sidebar for user selection
        unique_users = sorted(df['User'].dropna().unique())
        user_option = st.sidebar.radio("Select analysis option (For Word Cloud):", ["All Users", "Individual Users"])

        # Sidebar filter for date range
        st.sidebar.header("Filter by Date Range")
        start_date = st.sidebar.date_input("Start Date", df['Date'].min().date())
        end_date = st.sidebar.date_input("End Date", df['Date'].max().date())

        # Apply the date filter
        if start_date > end_date:
            st.sidebar.error("Start Date cannot be after End Date!")
        else:
            filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
            st.success(f"Filtered {len(filtered_df)} messages from {start_date} to {end_date}.")

            # Replace `df` in subsequent analyses with `filtered_df`
            df = filtered_df

        
        # Function to calculate message count per user
        @st.cache_data
        def calculate_message_count_per_user(df):
            return df['User'].value_counts().reset_index(name='Message Count').rename(columns={'index': 'User'})

        # Generate Donut Chart for User Messages
        # if user_option == "All Users":
        #     st.header("Messages per User")
        #     user_message_count = calculate_message_count_per_user(df)

        #     fig = px.pie(
        #         user_message_count, 
        #         values='Message Count', 
        #         names='User', 
        #         title='Message Distribution by User', 
        #         hole=0.5, 
        #         color_discrete_sequence=px.colors.qualitative.Set3
        #     )
        #     fig.update_traces(textinfo='percent+label')  # Show percentage and user labels
        #     st.plotly_chart(fig, use_container_width=True)

        # Interactive Donut Chart for User Messages
        # Add a dropdown for user selection
        st.header("Messages per User")

        user_message_count = calculate_message_count_per_user(df)
        fig = px.pie(
            user_message_count,
            values='Message Count',
            names='User',
            title='Message Distribution by User',
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_traces(textinfo='percent+label')

        # Display the pie chart
        st.plotly_chart(fig, use_container_width=True)

        # Dropdown for user selection
        selected_user = st.selectbox("Select a user to view details:", ["All"] + user_message_count['User'].tolist())

        if selected_user != "All":
            selected_user_data = df[df['User'] == selected_user]
            st.subheader(f"Detailed Analysis for {selected_user}")
            st.dataframe(selected_user_data)
            
            
        
        # Calculate emoji usage
        emoji_count_all, user_emoji_data = calculate_emoji_usage(df)

        # Most Used Emojis Across All Users
        st.header("Top Emojis Used Across All Users")

        if emoji_count_all:
            emoji_df_all = pd.DataFrame(emoji_count_all, columns=['Emoji', 'Count'])
            fig_all_emojis = px.bar(
                emoji_df_all,
                x='Emoji',
                y='Count',
                title='Top Emojis Used Across All Users',
                labels={'Emoji': 'Emoji', 'Count': 'Usage Count'},
                color_discrete_sequence=["#636EFA"],
            )
            st.plotly_chart(fig_all_emojis, use_container_width=True)
        else:
            st.write("No emojis found in the dataset.")


        # Emoji Analysis for Each User
        st.header("Emoji Analysis for Each User")

        selected_user_emoji = st.selectbox("Select a user:", ["All"] + list(user_emoji_data.keys()))

        if selected_user_emoji == "All":
            st.write("Viewing emoji data for all users. See the chart above.")
        else:
            user_emoji_count = user_emoji_data[selected_user_emoji]
            if user_emoji_count:
                emoji_df_user = pd.DataFrame(user_emoji_count, columns=['Emoji', 'Count'])
                fig_user_emojis = px.bar(
                    emoji_df_user,
                    x='Emoji',
                    y='Count',
                    title=f"Top Emojis Used by {selected_user_emoji}",
                    labels={'Emoji': 'Emoji', 'Count': 'Usage Count'},
                    color_discrete_sequence=["#EF553B"],
                )
                st.plotly_chart(fig_user_emojis, use_container_width=True)
            else:
                st.write(f"No emojis found for {selected_user_emoji}.")


        
            
        
        # Functions to create time-based data for plotting
        @st.cache_data
        def get_hourwise_activity(df):
            # Convert 'Time' to datetime with flexible parsing
            df['Hour'] = pd.to_datetime(df['Time'], format='%I:%M %p', errors='coerce').dt.hour
           
            return df.groupby(['User', 'Hour']).size().reset_index(name='Message Count')

        @st.cache_data
        def get_monthwise_activity(df):
            df['Month'] = df['Date'].dt.to_period('M').astype(str)
            return df.groupby('Month').size().reset_index(name='Message Count')

        @st.cache_data
        def get_weekday_activity(df):
            df['Weekday'] = df['Date'].dt.day_name()
            return df.groupby('Weekday').size().reindex(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            ).reset_index(name='Message Count')


        # Hourly message activity graph
        # st.header("Hourly Message Activity")
        # hourwise_data = get_hourwise_activity(df)

        # # Sidebar selection for user
        # selected_user_for_hour = st.sidebar.selectbox("Choose a user for hourly activity:", ['All'] + unique_users)

        # if selected_user_for_hour == 'All':
        #     hourly_fig = px.bar(
        #         hourwise_data.groupby('Hour')['Message Count'].sum().reset_index(),
        #         x='Hour',
        #         y='Message Count',
        #         title='Messages per Hour (All Users)',
        #         labels={'Hour': 'Hour of the Day', 'Message Count': 'Number of Messages'},
        #         color_discrete_sequence=["#AB63FA"]
        #     )
        # else:
        #     user_hourwise_data = hourwise_data[hourwise_data['User'] == selected_user_for_hour]
        #     hourly_fig = px.bar(
        #         user_hourwise_data,
        #         x='Hour',
        #         y='Message Count',
        #         title=f'Messages per Hour ({selected_user_for_hour})',
        #         labels={'Hour': 'Hour of the Day', 'Message Count': 'Number of Messages'},
        #         color_discrete_sequence=["#FFA15A"]
        #     )
        # st.plotly_chart(hourly_fig, use_container_width=True)
        
        
        # Hourly Activity with User and Hour Selection
        # st.header("Hourly Message Activity")

        # hourwise_data = get_hourwise_activity(df)
        # selected_user_for_hour = st.sidebar.selectbox("Choose a user (For Hourly data):", ["All"] + unique_users)

        # if selected_user_for_hour == "All":
        #     hourly_fig = px.bar(
        #         hourwise_data.groupby('Hour')['Message Count'].sum().reset_index(),
        #         x='Hour',
        #         y='Message Count',
        #         title='Messages per Hour (All Users)',
        #         labels={'Hour': 'Hour of the Day', 'Message Count': 'Number of Messages'},
        #         color_discrete_sequence=["#AB63FA"],
        #     )
        # else:
        #     user_hourwise_data = hourwise_data[hourwise_data['User'] == selected_user_for_hour]
        #     hourly_fig = px.bar(
        #         user_hourwise_data,
        #         x='Hour',
        #         y='Message Count',
        #         title=f'Messages per Hour ({selected_user_for_hour})',
        #         labels={'Hour': 'Hour of the Day', 'Message Count': 'Number of Messages'},
        #         color_discrete_sequence=["#FFA15A"],
        #     )

        # st.plotly_chart(hourly_fig, use_container_width=True)

        # # Hour range filter
        # hour_range = st.slider("Select hour range:", 0, 23, (0, 23), step=1)
        # filtered_df = df[df['Time'].apply(lambda x: int(x.split(":")[0]) in range(hour_range[0], hour_range[1] + 1))]

        # st.subheader(f"Messages between {hour_range[0]}:00 and {hour_range[1]}:00")
        # st.dataframe(filtered_df)
        
        
        
        # Hourly Activity with User and Hour Selection
        st.header("Hourly Message Activity")
        
        # Hourly Data Preparation
        hourwise_data = get_hourwise_activity(df)
        selected_user_for_hour = st.sidebar.selectbox("Choose a user (For Hourly data):", ["All"] + unique_users)
        
        # Date Selection
        st.sidebar.subheader("Filter by Date")
        date_range = st.sidebar.date_input("Select a date or range:", [])
        
        if isinstance(date_range, list) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_by_date = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        elif isinstance(date_range, list) and len(date_range) == 1:
            filtered_by_date = df[df['Date'] == date_range[0]]
        else:
            filtered_by_date = df  # No date filter applied
        
        # User-specific Hourly Data
        if selected_user_for_hour == "All":
            hourly_fig = px.bar(
                hourwise_data.groupby('Hour')['Message Count'].sum().reset_index(),
                x='Hour',
                y='Message Count',
                title='Messages per Hour (All Users)',
                labels={'Hour': 'Hour of the Day', 'Message Count': 'Number of Messages'},
                color_discrete_sequence=["#AB63FA"],
            )
        else:
            user_hourwise_data = hourwise_data[hourwise_data['User'] == selected_user_for_hour]
            hourly_fig = px.bar(
                user_hourwise_data,
                x='Hour',
                y='Message Count',
                title=f'Messages per Hour ({selected_user_for_hour})',
                labels={'Hour': 'Hour of the Day', 'Message Count': 'Number of Messages'},
                color_discrete_sequence=["#FFA15A"],
            )
        
        st.plotly_chart(hourly_fig, use_container_width=True)
        
        # Hour Range Filter
        hour_range = st.slider("Select hour range:", 0, 23, (0, 23), step=1)
        
        # Filter by Hour and Date
        filtered_by_hour_and_date = filtered_by_date[
            filtered_by_date['Time'].apply(lambda x: int(x.split(":")[0]) in range(hour_range[0], hour_range[1] + 1))
        ]
        
        st.subheader(f"Messages between {hour_range[0]}:00 and {hour_range[1]}:00 on Selected Dates")
        st.dataframe(filtered_by_hour_and_date)
        
        
        
        
        

        
        # Monthly graph
        st.header("Monthly Message Activity")
        monthwise_data = get_monthwise_activity(df)
        fig_monthly = px.bar(
            monthwise_data, 
            x='Month', 
            y='Message Count', 
            title='Messages per Month', 
            labels={'Month': 'Month', 'Message Count': 'Number of Messages'},
            color_discrete_sequence=["#EF553B"]
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Weekday graph
        st.header("Weekday Message Activity")
        weekday_data = get_weekday_activity(df)
        fig_weekday = px.bar(
            weekday_data, 
            x='Weekday', 
            y='Message Count', 
            title='Messages by Weekday', 
            labels={'Weekday': 'Weekday', 'Message Count': 'Number of Messages'},
            color_discrete_sequence=["#00CC96"]
        )
        st.plotly_chart(fig_weekday, use_container_width=True)
        

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
            selected_user = st.sidebar.selectbox("Choose a user (For Word clouud):", unique_users)
            user_text = " ".join(df[df['User'] == selected_user]['Message'].dropna())
            wordcloud = generate_wordcloud(user_text)
            st.image(wordcloud.to_image(), caption=f"Word Cloud for {selected_user}", use_container_width=True)

            # Display individual user stats
            st.header(f"Detailed Analysis for {selected_user}")
            stats = calculate_user_stats(df, selected_user)
            for key, value in stats.items():
                st.write(f"**{key}:** {value}")

        
        # sentiment analysis 
        df = perform_sentiment_analysis(df)

        # Sentiment Summary
        st.header("Sentiment Analysis Summary")

        # Overall Sentiment Distribution
        sentiment_counts = df['Sentiment'].value_counts()
        fig_sentiment = px.pie(
            sentiment_counts,
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Overall Sentiment Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

        # Sentiment Analysis by User
        st.header("Sentiment Analysis by User")
        user_sentiment_option = st.sidebar.selectbox("Select a user (For Sentiment Data):", ["All"] + unique_users)

        if user_sentiment_option == "All":
            sentiment_by_user = df.groupby('Sentiment').size().reset_index(name='Count')
            fig_user_sentiment = px.bar(
                sentiment_by_user,
                x='Sentiment',
                y='Count',
                title='Sentiment Distribution Across All Users',
                color='Sentiment',
                color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"]
            )
            st.plotly_chart(fig_user_sentiment, use_container_width=True)
        else:
            user_data = df[df['User'] == user_sentiment_option]
            sentiment_user_specific = user_data['Sentiment'].value_counts().reset_index(name='Count').rename(columns={'index': 'Sentiment'})
            fig_user_specific_sentiment = px.bar(
                sentiment_user_specific,
                x='Sentiment',
                y='Count',
                title=f"Sentiment Distribution for {user_sentiment_option}",
                color='Sentiment',
                color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"]
            )
            st.plotly_chart(fig_user_specific_sentiment, use_container_width=True)

        # Top Positive and Negative Messages
        st.header("Top Positive and Negative Messages")
        top_positive_messages = df[df['Polarity'] > 0].sort_values(by='Polarity', ascending=False).head(5)
        top_negative_messages = df[df['Polarity'] < 0].sort_values(by='Polarity').head(5)

        st.subheader("Top 5 Positive Messages")
        st.dataframe(top_positive_messages[['Date', 'User', 'Message', 'Polarity']])

        st.subheader("Top 5 Negative Messages")
        st.dataframe(top_negative_messages[['Date', 'User', 'Message', 'Polarity']])
        
        
        
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
