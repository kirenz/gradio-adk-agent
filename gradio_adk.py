"""
Simple Time Agent with Google ADK and Gradio.

A single-file example demonstrating how to build an agent with tools
that can tell you the current time in different cities.
"""

import asyncio
import datetime
import gradio as gr
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import InMemoryRunner

# Load environment variables
load_dotenv()


# ============================================================================
# Tool Functions
# ============================================================================

def calculate(expression: str) -> dict:
    """
    Evaluates a mathematical expression safely.

    Args:
        expression: A mathematical expression as string (e.g., "2 + 2", "sqrt(16)", "sin(pi/2)")

    Returns:
        Dictionary with status and result
    """
    try:
        import math
        import re

        # Only allow safe mathematical operations
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("__")
        }
        # Add basic operators
        allowed_names["abs"] = abs
        allowed_names["round"] = round

        # Security check: only allow safe characters
        if not re.match(r'^[0-9+\-*/().,\s\w]+$', expression):
            return {"status": "error", "message": "Invalid characters in expression"}

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"status": "success", "result": result, "expression": expression}
    except Exception as e:
        return {"status": "error", "message": str(e), "expression": expression}


def get_current_datetime() -> dict:
    """Returns the current date and time."""
    now = datetime.datetime.now()
    return {
        "status": "success",
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": now.strftime("%A")
    }


# ============================================================================
# Agent and Runner Setup
# ============================================================================

root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='helper_agent',
    description="A helpful assistant with calculator and datetime tools.",
    instruction="""
    You are a helpful assistant with access to useful tools:
    - calculate: For mathematical calculations (supports basic math, trigonometry, etc.)
    - get_current_datetime: For getting the current date and time

    Use these tools when needed to answer user questions.
    Answer in the same language as the user's question.
    Be friendly and concise in your response.
    """,
    tools=[calculate, get_current_datetime],
)

runner = InMemoryRunner(agent=root_agent, app_name='time_app')

# Global session ID for maintaining conversation history
SESSION_ID = None


# ============================================================================
# Chat Processing Function
# ============================================================================

async def chat_with_agent_async(message: str) -> str:
    """Sends a message to the agent and returns the response."""
    global SESSION_ID

    # Create session only once, reuse for all subsequent messages
    if SESSION_ID is None:
        session = await runner.session_service.create_session(
            user_id='gradio_user',
            app_name='time_app'
        )
        SESSION_ID = session.id

    # Create content object
    content = types.Content(
        role='user',
        parts=[types.Part(text=message)]
    )

    # Run agent and collect response
    response_text = ""
    events_async = runner.run_async(
        user_id='gradio_user',
        session_id=SESSION_ID,
        new_message=content
    )

    async for event in events_async:
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text
            break

    return response_text


def chat_with_agent(message: str, history: list) -> str:
    """Synchronous wrapper for Gradio."""
    try:
        response = asyncio.run(chat_with_agent_async(message))
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def reset_session():
    """Resets the session to start a new conversation."""
    global SESSION_ID
    SESSION_ID = None
    return []


# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="Google ADK Helper Agent") as demo:

    gr.Markdown("# Helper Agent with Google ADK")
    gr.Markdown("Ask the agent to help with calculations or get the current date/time!")

    chatbot = gr.Chatbot(label="Chat", height=400)

    msg = gr.Textbox(
        label="Your Message",
        placeholder="e.g., 'Calculate 15 * 23' or 'What's today's date?'",
        lines=1
    )

    clear = gr.Button("Clear Chat")

    # Examples
    gr.Examples(
        examples=[
            ["Calculate 15 * 23 + 45"],
            ["What is the square root of 144?"],
            ["What's today's date?"],
            ["Calculate sin(pi/2)"],
            ["What day of the week is it?"],
        ],
        inputs=msg
    )

    def respond(message: str, chat_history: list):
        """Processes the message and updates the chat."""
        bot_response = chat_with_agent(message, chat_history)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_response})
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(reset_session, None, chatbot)


if __name__ == "__main__":
    demo.launch()
