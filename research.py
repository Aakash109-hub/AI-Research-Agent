import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("🚨 Error: GROQ_API_KEY not found in .env file!")
    st.stop()

# Initialize Research Agent
Research_Agent = Agent(
    model=Groq(id="deepseek-r1-distill-llama-70b", api_key=groq_api_key),  # Use a smaller model
    tools=[DuckDuckGo(), Newspaper4k()],  # Reduce fetched sources
    description="You are a senior NYT researcher writing an article on a topic.",
    instructions=[
        "Find and summarize the top 3 sources for a given topic.",
        "Ignore broken links and unavailable pages.",
        "Generate a research summary in clear, structured paragraphs.",
    ],
    markdown=True,
    show_tool_calls=False,
    add_datetime_to_instructions=True
)

# Streamlit UI
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Title and Header
st.title("🧐 SmartInsight AI: Discover, Analyze, Conquer")
st.markdown("---")
st.subheader("Enter a topic, and our AI will generate a research summary.")

# User input for research topic
topic = st.text_input("🔍 Enter Research Topic:", placeholder="E.g., The Future of Quantum Computing", key="topic_input")

# Button to trigger research
if st.button("🚀 Generate Research", key="generate_research_button"):
    if not topic.strip():
        st.warning("⚠️ Please enter a valid topic.")
    else:
        try:
            with st.spinner("🔄 Fetching AI-generated research..."):
                # 🔹 Create a structured prompt
                research_prompt = (f"""
                    Conduct a research study on the topic: {topic}
                    - Gather insights from at least three reputable sources.
                    - Summarize the key findings concisely.
                    - Structure the response in a well-organized article format.
                    """
                )

                # ✅ Use `run()` to directly get response
                response = Research_Agent.run(research_prompt)

            # 🔹 Display the result
            st.markdown("---")
            st.subheader("📜 Research Summary")
            st.markdown(response.content)  # Display AI-generated research
            st.markdown("---")
            st.success("✅ Research summary generated successfully!")

        except Exception as error:
            st.error(f"🚨 An error occurred: {error}")


