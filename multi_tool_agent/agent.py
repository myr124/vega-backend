import os

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass


from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool

from .util import load_instruction_from_file

# --- Sub Agent 2: Summarizer ---
summarizer_agent = LlmAgent(
    name="VideoSummarizer",
    model="gemini-2.5-flash-lite",
    instruction=load_instruction_from_file("./instructions/transcript_summarizer.txt"),
    description="Creates concise summaries from video transcripts to reduce token usage",
    output_key="video_summary",  # Save result to state
)

summarize_tool = AgentTool(agent=summarizer_agent)

# --- Sub Agent 1: Transcriber ---
transcriber_agent = LlmAgent(
    name="VideoTranscriber",
    model="gemini-2.0-flash-lite",
    instruction=load_instruction_from_file("./instructions/video_transcriber.txt"),
    description="Transcribes audio from video files into clean, formatted text",
    tools=[summarize_tool],
    output_key="video_transcript",  # Save result to state
)

# --- Create research agents with different personalities (reduced set for testing) ---
research_agents = []

# Map archetypes to their specialist categories
archetype_to_category = {
    "shopping": "Shopping",
    "music": "Music",
    "movies_tv": "Movies & Tv",
    "gaming": "Gaming",
    "news": "news",
    "sports": "Sports",
    "learning": "Learning",
    "fashion_beauty": "Fashion & Beauty",
}

personality_archetypes = [
    "shopping",
    "music",
    "movies_tv",
    "gaming",
    "news",
    "sports",
    "learning",
    "fashion_beauty",
]  # 8 categories
interest_levels = ["beginner", "intermediate", "expert"]

for archetype in personality_archetypes:
    category = archetype_to_category[archetype]
    for level in interest_levels:
        agent_name = f"{archetype}_{level}_reviewer"
        instruction_text = (
            load_instruction_from_file("./instructions/enjoyer_instruction.txt")
            .replace("{level}", level)
            .replace("{category}", category)
        )
        agent = LlmAgent(
            name=agent_name,
            model="gemini-2.0-flash-lite",
            instruction=instruction_text,
            description=f"{archetype} reviewer with {level} level perspective in {category}",
            output_key=f"{archetype}_{level}_review",
        )
        research_agents.append(agent)

# --- Parallel Research Agent (for reviewer personalities) ---
parallel_research_agent = ParallelAgent(
    name="parallel_research_agent",
    sub_agents=research_agents,  # 24 agents (8 categories × 3 skill levels)
    description="Runs multiple reviewer personalities in parallel",
)


class outputSchema(BaseModel):
    output: str = Field(
        description=load_instruction_from_file("./instructions/outputschema.txt")
    )


# --- Merger Agent ---
merger_agent = LlmAgent(
    name="merger_agent",
    model="gemini-2.5-flash-lite",
    instruction=load_instruction_from_file("./instructions/synthesis_prompt.txt"),
    description="Merges and synthesizes outputs from multiple reviewer agents into a cohesive final output.",
    output_schema=outputSchema,
    output_key="final_summary",
)

# --- Sequential Pipeline (Following ADK Documentation Pattern) ---
# This matches the exact pattern from the official ADK docs for ParallelAgent usage
sequential_pipeline_agent = SequentialAgent(
    name="VideoAnalysisPipeline",
    sub_agents=[
        transcriber_agent,  # Phase 1: Video processing (transcript + summary)
        parallel_research_agent,  # Phase 2: Parallel reviewer analysis (15 agents)
        merger_agent,  # Phase 3: Synthesis of all results
    ],
    description="Coordinates video processing, parallel research, and synthesis following ADK best practices.",
)

# --- Root Agent (Following ADK Documentation) ---
# The SequentialAgent is the root agent, as shown in the official docs
root_agent = sequential_pipeline_agent
