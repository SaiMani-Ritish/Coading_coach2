import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import google.generativeai as genai
from difflib import get_close_matches
import re
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini AI
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit page config
st.set_page_config(
    page_title="LeetCode Problem Tracker",
    page_icon="💻",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_problems(csv_file="leetcode_question.csv"):
    """Load problems from CSV file"""
    try:
        return pd.read_csv(csv_file)
    except FileNotFoundError:
        st.error(f"❌ {csv_file} not found. Please ensure the file exists in the app directory.")
        return pd.DataFrame()

def append_to_history(new_entry, filename="all_attempts.json"):
    """Append new attempt to history"""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(new_entry)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def get_problem_link_by_title(title, df):
    """Get problem link by matching title"""
    if df.empty:
        return "https://leetcode.com", title, False
    
    title = title.lower().strip()
    df["normalized_title"] = df["Title"].str.lower().str.strip()
    
    titles = df["normalized_title"].tolist()
    close = get_close_matches(title, titles, n=1, cutoff=0.6)
    
    if close:
        match_row = df[df["normalized_title"] == close[0]].iloc[0]
        return match_row["Leetcode Question Link"], match_row["Title"], True
    
    return "https://leetcode.com", title, False

def check_revision_needed(attempts, df):
    """Check if any problem needs revision (solved 7 days ago)"""
    today = datetime.now().date()
    for attempt in attempts:
        if attempt["Completed"] == "yes":
            try:
                date_solved = datetime.strptime(attempt["date_attempted"], "%Y-%m-%d").date()
                if (today - date_solved).days == 7:
                    link = attempt.get("Leetcode Question Link", "").strip()
                    if not link:
                        link, matched_title, found = get_problem_link_by_title(attempt["Title"], df)
                        attempt["Title"] = matched_title
                        attempt["Leetcode Question Link"] = link
                    return attempt
            except ValueError:
                continue
    return None

def pick_problem_with_ai(df, prev_title, prev_difficulty, recent_tags, completed, date_attempted, all_attempts):
    """Use AI to pick the next problem"""
    if not os.getenv("GOOGLE_API_KEY"):
        return None, False
    
    # Check for revision priority
    revision_problem = check_revision_needed(all_attempts, df)
    if revision_problem:
        return json.dumps({
            "Title": revision_problem["Title"],
            "Difficulty": revision_problem["Difficulty"],
            "Link": revision_problem.get("Leetcode Question Link", "https://leetcode.com"),
            "Reason": "This problem is due for revision as it was solved exactly 7 days ago."
        }), True

    prompt = f"""
You are an AI tutor designed to help a student practice Data Structures and Algorithms (DSA) on LeetCode.

The student recently attempted the LeetCode problem titled **"{prev_title}"** with difficulty **{prev_difficulty}**.
Completion status: **{completed}**.
Date: {date_attempted}.

### Guidelines for Selecting the Next Problem:
- If the last problem was **Easy** and **not completed**, suggest the **same or easier**.
- If **Medium** and **not completed**, suggest an **Easy or Medium** problem.
- If **Hard** and **not completed**, suggest a **Medium**.
- If completed, increase or maintain challenge level, staying within similar or varied topics.
- Avoid repeating tags: {recent_tags}

📚 Summary of Recent Attempts:
""" + "\n".join([
        f"- {a['Title']} ({a['Difficulty']}): {'Completed' if a['Completed'] == 'yes' else 'Skipped'} on {a['date_attempted']}"
        for a in all_attempts[-5:]
    ]) + """

🎯 Return result as:
{
    "Title": "<problem title>",
    "Difficulty": "<difficulty>",
    "Link": "<Leetcode link>",
    "Reason": "<1-sentence reason>"
}
ONLY return valid JSON.
"""

    try:
        response = model.generate_content(prompt)
        return response.text, False
    except Exception as e:
        st.error(f"AI API Error: {str(e)}")
        return None, False

def save_selected_problem(problem_title, problem_link, prev_difficulty, recent_tags, user_behavior, reason, is_revision=False, completed='no'):
    """Save selected problem to JSON"""
    data = {
        "Title": problem_title,
        "Leetcode Question Link": problem_link,
        "Previous Difficulty": prev_difficulty,
        "Recent Tags": recent_tags,
        "User Behavior": user_behavior,
        "Reason": reason
    }
    if is_revision:
        data["Tag"] = "revision"
    if completed == 'no':
        data["Tag"] = data.get("Tag", "") + " not Complete"

    with open("selected_problem.json", "w") as f:
        json.dump(data, f, indent=4)

def load_attempts_history():
    """Load attempts history"""
    if os.path.exists("all_attempts.json"):
        try:
            with open("all_attempts.json", "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 LeetCode Problem Tracker</h1>', unsafe_allow_html=True)
    
    # Check for required files and API key
    missing_requirements = []
    if not os.path.exists("leetcode_question.csv"):
        missing_requirements.append("leetcode_question.csv")
    if not os.getenv("GOOGLE_API_KEY"):
        missing_requirements.append("GOOGLE_API_KEY in .env file")
    if not os.getenv("TO_EMAIL"):
        missing_requirements.append("TO_EMAIL in .env file")
    
    if missing_requirements:
        st.error(f"❌ Missing requirements: {', '.join(missing_requirements)}")
        st.info("Please ensure all required files and environment variables are set up.")
        return
    
    # Load data
    df = load_problems()
    all_attempts = load_attempts_history()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Submit Attempt", "View History", "Manual Email"])
    
    if page == "Submit Attempt":
        st.header("📝 Submit Your Previous Attempt")
        
        with st.form("problem_attempt_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Problem Title*", placeholder="e.g., Two Sum")
                difficulty = st.selectbox("Difficulty*", ["Easy", "Medium", "Hard"])
                time_taken = st.text_input("Time Taken", placeholder="e.g., 30 mins")
            
            with col2:
                completed = st.selectbox("Did you complete it?*", ["yes", "no"])
                tags = st.text_input("Tags (comma-separated)", placeholder="e.g., array, hash-table")
                date_attempted = st.date_input("Date Attempted*", value=datetime.now().date())
            
            submitted = st.form_submit_button("🎯 Get Next Problem", use_container_width=True)
            
            if submitted:
                if not title or not difficulty:
                    st.error("❌ Please fill in all required fields (marked with *)")
                else:
                    with st.spinner("Processing your attempt and finding the next problem..."):
                        # Prepare the attempt data
                        previous_attempt = {
                            "Title": title,
                            "Difficulty": difficulty,
                            "Time Taken": time_taken,
                            "Completed": completed,
                            "Tags": [tag.strip() for tag in tags.split(",")] if tags else [],
                            "date_attempted": date_attempted.strftime("%Y-%m-%d")
                        }
                        
                        # Save to history
                        append_to_history(previous_attempt)
                        
                        # Get all attempts for AI context
                        all_attempts = load_attempts_history()
                        
                        # Get AI recommendation
                        ai_response, is_revision = pick_problem_with_ai(
                            df,
                            prev_title=previous_attempt["Title"],
                            prev_difficulty=previous_attempt["Difficulty"],
                            recent_tags=previous_attempt["Tags"],
                            completed=previous_attempt["Completed"],
                            date_attempted=previous_attempt["date_attempted"],
                            all_attempts=all_attempts
                        )
                        
                        if ai_response:
                            try:
                                # Parse AI response
                                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                                if json_match:
                                    parsed = json.loads(json_match.group())
                                else:
                                    raise ValueError("No valid JSON found in AI response")
                                
                                problem_title = parsed["Title"]
                                problem_link = parsed["Link"]
                                reason = parsed["Reason"]
                                user_behavior = "skipped" if previous_attempt["Completed"] == "no" else "completed"
                                
                                # Save selected problem
                                save_selected_problem(
                                    problem_title,
                                    problem_link,
                                    previous_attempt["Difficulty"],
                                    previous_attempt["Tags"],
                                    user_behavior,
                                    reason,
                                    is_revision,
                                    completed=previous_attempt["Completed"]
                                )
                                
                                # Display success
                                st.markdown('<div class="success-box">✅ Attempt saved successfully!</div>', unsafe_allow_html=True)
                                
                                # Display next problem
                                st.subheader("🎯 Your Next Problem")
                                
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"**Title:** {problem_title}")
                                    st.markdown(f"**Difficulty:** {parsed['Difficulty']}")
                                    st.markdown(f"**Reason:** {reason}")
                                    if is_revision:
                                        st.markdown("🔄 **This is a revision problem!**")
                                
                                with col2:
                                    st.markdown(f"[🔗 Solve Problem]({problem_link})")
                                
                                # Option to send email
                                if st.button("📧 Send Email Now", use_container_width=True):
                                    with st.spinner("Sending email..."):
                                        try:
                                            result = subprocess.run(["python", "agent2_send_email.py"], 
                                                                 capture_output=True, text=True, timeout=30)
                                            if result.returncode == 0:
                                                st.success("📧 Email sent successfully!")
                                            else:
                                                st.error(f"Email sending failed: {result.stderr}")
                                        except subprocess.TimeoutExpired:
                                            st.error("Email sending timed out. Please check your email configuration.")
                                        except Exception as e:
                                            st.error(f"Error sending email: {str(e)}")
                                
                            except Exception as e:
                                st.error(f"❌ Failed to parse AI response: {str(e)}")
                                st.code(ai_response)
                        else:
                            st.error("❌ Failed to get AI recommendation. Please check your API key and try again.")
    
    elif page == "View History":
        st.header("📊 Your Problem History")
        
        if all_attempts:
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_attempts = len(all_attempts)
            completed_count = sum(1 for attempt in all_attempts if attempt["Completed"] == "yes")
            completion_rate = (completed_count / total_attempts) * 100 if total_attempts > 0 else 0
            
            with col1:
                st.metric("Total Attempts", total_attempts)
            with col2:
                st.metric("Completed", completed_count)
            with col3:
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
            with col4:
                recent_streak = 0
                for attempt in reversed(all_attempts):
                    if attempt["Completed"] == "yes":
                        recent_streak += 1
                    else:
                        break
                st.metric("Current Streak", recent_streak)
            
            # Display recent attempts
            st.subheader("Recent Attempts")
            recent_attempts = all_attempts[-10:][::-1]  # Last 10, reversed
            
            for i, attempt in enumerate(recent_attempts):
                with st.expander(f"{attempt['Title']} - {attempt['Difficulty']} ({'✅' if attempt['Completed'] == 'yes' else '❌'})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Date:** {attempt['date_attempted']}")
                        st.write(f"**Time Taken:** {attempt.get('Time Taken', 'N/A')}")
                    with col2:
                        st.write(f"**Status:** {'Completed' if attempt['Completed'] == 'yes' else 'Skipped'}")
                        if attempt.get('Tags'):
                            st.write(f"**Tags:** {', '.join(attempt['Tags'])}")
        else:
            st.info("No attempts recorded yet. Submit your first attempt to get started!")
    
    elif page == "Manual Email":
        st.header("📧 Send Manual Email")
        st.info("This will send an email based on the last selected problem in selected_problem.json")
        
        if os.path.exists("selected_problem.json"):
            try:
                with open("selected_problem.json", "r") as f:
                    problem_data = json.load(f)
                
                st.subheader("Current Problem Data")
                st.json(problem_data)
                
                if st.button("📧 Send Email", use_container_width=True):
                    with st.spinner("Sending email..."):
                        try:
                            result = subprocess.run(["python", "agent2_send_email.py"], 
                                                 capture_output=True, text=True, timeout=30)
                            if result.returncode == 0:
                                st.success("📧 Email sent successfully!")
                            else:
                                st.error(f"Email sending failed: {result.stderr}")
                        except subprocess.TimeoutExpired:
                            st.error("Email sending timed out. Please check your email configuration.")
                        except Exception as e:
                            st.error(f"Error sending email: {str(e)}")
            except json.JSONDecodeError:
                st.error("Invalid JSON in selected_problem.json")
        else:
            st.warning("No selected problem found. Please submit an attempt first.")

if __name__ == "__main__":
    main()