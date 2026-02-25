from pathlib import Path

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

import asyncio
import streamlit as st
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from transportAgent.agent import root_agent  # import from same package

# --- Page Config ---
st.set_page_config(
    page_title="Transport Planner Agent",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Transport Planner Agent")
st.markdown("Plan your travel with AI-powered transport recommendations.")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_service" not in st.session_state:
    st.session_state.session_service = InMemorySessionService()
if "session_created" not in st.session_state:
    st.session_state.session_created = False

APP_NAME = "transport_streamlit_app"
USER_ID = "streamlit_user"
SESSION_ID = "streamlit_session"


async def create_session():
    """Create ADK session if not exists."""
    if not st.session_state.session_created:
        await st.session_state.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        st.session_state.session_created = True


async def run_agent(query: str) -> str:
    """Run the agent and return the response."""
    await create_session()

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=st.session_state.session_service
    )

    content = types.Content(role='user', parts=[types.Part(text=query)])
    response_text = ""

    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.is_final_response():
            # keep overwriting so the last final response wins (handles
            # workflows with multiple sub-agents)
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text
            # do not break; let all agents run so we capture the ultimate answer

    return response_text


def get_response(query: str) -> str:
    """Wrapper to run async function."""
    return asyncio.run(run_agent(query))


# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Trip Planner Form ---
with st.form("trip_form"):
    st.subheader("Plan a journey")
    start = st.text_input("Starting location")
    dest = st.text_input("Destination")
    mode = st.selectbox(
        "Preferred mode of transport (optional)",
        ["All", "MRT/Train", "Bus", "Taxi", "Cycling", "Walking"],
    )
    submitted = st.form_submit_button("Plan trip")
    if submitted:
        # build a query for the agent
        prompt = f"Plan a trip from {start} to {dest}."
        if mode and mode != "All":
            prompt += f" Prefer mode: {mode}."
        # Add user message to history and fetch response immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Planning your transport..."):
                response = get_response(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Chat Input ---
if prompt := st.chat_input("Or ask a question directly... (e.g., 'What's the fastest way to travel from Jurong to Marina Bay?')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Planning your transport..."):
            response = get_response(prompt)
            st.markdown(response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.markdown("""
    This transport planner uses multiple AI agents:
    - 🔍 **Research Agent** - Finds transport options
    - 💰 **Budget Agent** - Cost estimates
    - ⏰ **Time Agent** - Time-efficient routes
    """)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_created = False
        st.rerun()