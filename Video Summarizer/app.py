import streamlit as st 
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from google.generativeai import upload_file,get_file
import google.generativeai as genai
from agno.media import Video

import time
from pathlib import Path

import tempfile

from dotenv import load_dotenv
load_dotenv()

import os

API_KEY=os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Agent- Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

st.title("Phidata Video AI Summarizer Agent 🎥🎤🖬")
st.header("Powered by Gemini 2.0 Flash Exp")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash"),
        tools=[DuckDuckGoTools()],
        markdown=True,
    )

## Initialize the agent
multimodal_Agent=initialize_agent()

# File uploader
video_file = st.file_uploader(
    "Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video for AI analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name
        print(video_path)

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    # Upload and process video file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(2)
                        processed_video = get_file(processed_video.name)

                    # Create a video representation compatible with Agent
                    # gemini_video_url = processed_video.uri  # or possibly processed_video.name, depending on what agno expects

                    # Prompt generation for analysis
                    analysis_prompt = f"""
                    Analyze the uploaded video for content and context.
                    Respond to the following query using video insights and supplementary web research:
                    {user_query}

                    Provide a detailed, user-friendly, and actionable response.
                    """
                    st.write("Submitting video to AI agent...")
                    # AI agent processing
                    # print(multimodal_Agent.run.__annotations__)
                    response: RunResponse = multimodal_Agent.run(analysis_prompt, videos=[Video(content=processed_video)])
                    st.write("Agent response received.")

                # Display the result
                st.subheader("Analysis Result")
                # st.markdown(response.message)
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                # Clean up temporary video file
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to begin analysis.")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# import time
# from pathlib import Path
# from agno.agent import Agent
# from agno.media import Video
# from agno.models.google import Gemini
# from google.generativeai import upload_file, get_file
# import google.generativeai as genai

# import os

# API_KEY=os.getenv("GEMINI_API_KEY")
# if API_KEY:
#     genai.configure(api_key=API_KEY)
    
# model = Gemini(id="gemini-2.0-flash-exp")
# agent = Agent(model=model, markdown=True)

# # Upload video file to Gemini
# video_path = Path(r"C:\Users\akani\Videos\Captures\FIFA 23 2023-12-16 11-21-55.mp4")
# video_file = upload_file(video_path)

# # Wait for processing
# while video_file.state.name == "PROCESSING":
#     time.sleep(2)
#     video_file = get_file(video_file.name)
# print("Gemini Video URI:", video_file.uri)

# agent.print_response(
#     "Tell me about this video",
#     videos=[Video(url=video_file.uri)],
# )