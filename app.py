import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 1. OPEN THE VAULT ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🚨 CRITICAL ERROR: The API Key is missing! Python cannot find your .env file.")
    st.stop()

# --- 2. PAGE SETUP ---
st.set_page_config(page_title="Auto Data Scientist", page_icon="📊", layout="wide")
st.title("📊 The Automated Data Scientist")
st.write("Upload a CSV. This AI Agent writes Python code in the background to analyze and visualize your data.")

# --- 3. DATA INGESTION ---
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Raw Data Preview:")
    st.dataframe(df.head())
    
    # Initialize the Brain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        api_key=GEMINI_API_KEY, 
        temperature=0.1 
    )
    
    # Initialize the Agent
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True 
    )
    
    # --- UPGRADE 1: ONE-CLICK EDA (SIDEBAR) ---
    st.sidebar.header("⚙️ Data Science Tools")
    st.sidebar.write("Generate a full profile of your dataset with one click.")
    
    if st.sidebar.button("Generate Instant EDA"):
        with st.spinner("Agent is profiling the dataset... ⏳"):
            eda_prompt = """
            You are a senior data scientist. Perform a rapid Exploratory Data Analysis (EDA).
            Provide a clean, bulleted summary containing:
            1. Total number of rows and columns.
            2. A list of any columns that contain missing (null) values.
            3. The total number of duplicate rows.
            4. A brief, 2-sentence hypothesis on what this dataset is about based on the column names.
            """
            try:
                eda_response = agent.invoke(eda_prompt)
                st.sidebar.success("EDA Complete!")
                st.sidebar.write(eda_response["output"])
            except Exception as e:
                st.sidebar.error(f"Error during EDA: {e}")

    # --- UPGRADE 2: GRAPHING ENGINE (MAIN AREA) ---
    st.divider()
    st.write("### 🤖 Ask the AI Agent")
    st.info("💡 **Pro Tip:** You can ask the AI to draw graphs! Try: *'Plot a bar chart of the top 5 highest prices.'*")
    
    user_query = st.text_input("Enter your command:")
    
    if st.button("Run Analysis"):
        if user_query:
            with st.spinner("Agent is writing code and generating output... ⏳"):
                
                # Cleanup old graphs before running new ones
                if os.path.exists("temp_chart.png"):
                    os.remove("temp_chart.png")
                
                # The Secret Graphing Instruction
                graphing_prompt = user_query + "\n\nCRITICAL INSTRUCTION: If the user asks for a graph or plot, you MUST save it to the current directory as 'temp_chart.png' using plt.savefig('temp_chart.png'). Do NOT use plt.show()."
                
                try:
                    response = agent.invoke(graphing_prompt)
                    st.success("Task Complete!")
                    st.write("### 📝 Verdict:")
                    st.write(response["output"])
                    
                    # Display the graph if the AI generated one
                    if os.path.exists("temp_chart.png"):
                        st.write("### 📈 Generated Visualization:")
                        st.image("temp_chart.png")
                        
                except Exception as e:
                    st.error(f"The Agent encountered an error: {e}")
        else:
            st.warning("Please give the Agent a task first!")