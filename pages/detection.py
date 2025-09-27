import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📧 Spam Email Detection App")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Check required columns
    if 'spam' not in df.columns or 'text' not in df.columns:
        st.error("❌ Dataset must have columns 'spam' and 'text'")
    else:
        st.subheader("🔍 Data Preview")
        st.write(df.head())

        # Features and labels
        X = df['text']
        y = df['spam']

        # Vectorization
        cv = CountVectorizer()
        X = cv.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Predictions & Accuracy
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"✅ Accuracy: {acc*100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Demo Messages
        st.sidebar.subheader("💬 Demo Messages")
        demo_msgs = [
            "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now.",
            "Hi John, are we still meeting for lunch tomorrow?",
            "URGENT! Your account has been compromised. Reset your password immediately.",
            "Don't forget the meeting at 3 PM today.",
            "Exclusive offer just for you! Limited time only."
        ]
        selected_demo = st.sidebar.selectbox("Choose a demo message", [""] + demo_msgs)

        st.subheader("✍ Test a Message")
        user_input = st.text_area("Enter a message to check", value=selected_demo)

        if st.button("🔎 Detect"):
            if user_input.strip() != "":
                input_vec = cv.transform([user_input])
                prediction = model.predict(input_vec)[0]
                st.success(f"Prediction: *{prediction}*")
            else:
                st.warning("⚠ Please enter a message or select a demo one.")
                