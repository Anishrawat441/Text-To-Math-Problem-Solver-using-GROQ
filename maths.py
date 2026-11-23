
import streamlit as st
import re

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import create_agent


# Load Environment

load_dotenv()


# Streamlit UI

st.title("Text To Math Problem Solver using Groq")


# Groq API Key

Groq_api = st.sidebar.text_input(
    "Please enter your Groq API Key",
    type="password"
)

if not Groq_api:
    st.info("Please enter your Groq API key.")
    st.stop()


# LLM

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=Groq_api
)


# Wikipedia Tool

Wikipedia_wrapper = WikipediaAPIWrapper()

Wikipedia_tool = Tool(
    name="wikipedia",
    func=Wikipedia_wrapper.run,
    description=(
        "Search Wikipedia for general information about people, "
        "places, concepts and other topics."
    )
)


# Calculator Tool


def math_tool_func(expression: str) -> str:
    """
    Calculate a mathematical expression.
    Example: 25 * 40
    """

    try:
        # Remove unnecessary spaces
        expression = expression.strip()

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as e:
        return f"Error calculating expression: {e}"


calculator = Tool(
    name="calculator",
    func=math_tool_func,
    description=(
        "Use this tool for arithmetic calculations. "
        "Input must be a mathematical expression such as "
        "'25 * 40', '100 / 5', '2 ** 10', or '(20 + 10) * 5'."
    )
)


# Reasoning Tool

prompt = """
You are a mathematical reasoning assistant.

Solve the user's mathematical problem logically.

Show the solution step-by-step and give the final answer clearly.

Question:
{question}

Answer:
"""

Prompt = PromptTemplate(
    template=prompt,
    input_variables=["question"]
)

chain = Prompt | model


def reasoning_tool_func(question: str) -> str:
    """
    Solve mathematical and logical reasoning problems.
    """

    response = chain.invoke({
        "question": question
    })

    return response.content


Reasoning = Tool(
    name="Reasoning_Tool",
    func=reasoning_tool_func,
    description=(
        "Use this tool for mathematical reasoning, algebra, "
        "word problems, logical problems and questions that "
        "require step-by-step reasoning."
    )
)


# Build Agent

tools = [
    Wikipedia_tool,
    calculator,
    Reasoning
]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
You are a helpful mathematical assistant.

Your job is to solve the user's mathematical questions.

Use the calculator tool for exact arithmetic calculations.

Use the Reasoning_Tool for:
- Algebra
- Word problems
- Logical reasoning
- Step-by-step mathematical problems

Use Wikipedia only when the user asks for general factual information.

Always provide a clear and understandable final answer.
"""
)


# Chat History


if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I am a Math Chatbot. "
                "Ask me any mathematical question."
            )
        }
    ]


# Display Chat History

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# User Input

Question = st.text_area(
    "Please ask your question:",
    placeholder="Example: Solve 2x + 5 = 15"
)


# Find Answer

if st.button("Find My Answer"):

    if not Question.strip():

        st.warning("Please enter your question.")

    else:

        # Save user question
        st.session_state.messages.append({
            "role": "user",
            "content": Question
        })

        # Display user question
        with st.chat_message("user"):
            st.write(Question)

        # Callback
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )

        # Run agent
        with st.spinner("Generating Response..."):

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": Question
                        }
                    ]
                },
                config={
                    "callbacks": [st_cb]
                }
            )

        # Extract final answer
        answer = response["messages"][-1].content

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # Display answer
        with st.chat_message("assistant"):
            st.write(answer)

        