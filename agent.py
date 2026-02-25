from pathlib import Path
from dotenv import load_dotenv

# Load .env from this agent's directory (optional configuration)
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

MODEL = "gemini-2.5-flash"

# Sub‑agent to gather locations from the user
location_input_agent = LlmAgent(
    model=MODEL,
    name="location_input_agent",
    description="Agent that collects and confirms the current location and destination.",
    instruction="""You are the location Input agent.

Greet the user warmly and introduce yourself as the Route Planner Agent
Explain that you can help to find the fastest route for various transport methods (e.g., MRT, BUS, TAXI, CYCLING, WALKING).
Ask the user to provide a current Location and a destination location
Once the user provides their location as well as the destination, extract the locations and confirm it with the user.
Store the location in your response so it can be passed to the next agent.""",
)

# Sub‑agent to research and recommend transport options
route_research_agent = LlmAgent(
    model=MODEL,
    name="route_research_agent",
    description="""Agent that researches transport options in Singapore and
recommends the best mode of travel.""",
    instruction="""You are a Route Research agent specialized in transportation planning.

Base all timing related considerations to the current time of the country of interest.

If you dont receive any information regarding cross country travel from the previous agent, take the current location and destination location provided by the previous agent, starting from the given current location, and use the google_search tool to gather:
- Latest route information and travel times for various transport methods (TRAIN/MRT, BUS, TAXI, WALK)
- Traffic conditions and delays
- Alternative routes and options
- Public transport schedules and availability

After gathering information, synthesize everything into a comprehensive, well-formatted report that includes:
1. **By Bus** - A route from the current location to the destination strictly by bus only
2. **By MRT/Train** - A route from the current location to the destination strictly by MRT/Train only
3. **By Taxi** - A route from the current location to the destination strictly by taxi
4. **By Cycling** - A route from the current location to the destination strictly by cycling/biking
5. **By Walking** - A route from the current location to the destination strictly by walking
6. **Fastest Route** - A route from the current location to the destination by all means of transport available in SINGAPORE while optimizing the best cost to time spent ratio.

Present the information in a clear, professional format that would be concise for a person who is in a rush.""",
    tools=[google_search],
)

# Sequential workflow that runs the two sub‑agents in order
transport_workflow_agent = SequentialAgent(
    name="transport_workflow_agent",
    description="Workflow that asks for locations then computes a transport recommendation.",
    sub_agents=[location_input_agent, route_research_agent],
)

# Root agent is simply the workflow pipeline.  The first sub-agent
# already handles greeting and location extraction, so using the workflow
# directly ensures all queries (including explicit requests like
# "What's the fastest way from A to B?") are processed.
root_agent = LlmAgent(
    model=MODEL,
    name="root_agent",
    description="Root agent that introduces route planning capabilities and delegates to workflow.",
    instruction="""You are a friendly Route Planning Assistant.

When the user first interacts with you:
1. Greet them warmly
2. Introduce yourself and explain that you can help them plan routes in Singapore
3. Mention that you have a specialized Route Planner agent that will help gather their destination and current location and provide detailed route information

After your introduction, transfer control to the transport_workflow_agent to begin the analysis process.""",
    sub_agents=[transport_workflow_agent],
)
