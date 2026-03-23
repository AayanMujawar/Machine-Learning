"""
Experiment No. : 5
Title: Implementation of a tool - Intelligent agent using Large language models.
Objectives:
1. To understand the concept of LLM-based agents
2. To implement tool usage within an agent framework
"""

# 1. Install required Python libraries.
# Run this command in your terminal before executing:
# pip install langchain langchain-groq python-dotenv

# 2. Import Required Modules.
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import tool, create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

def main():
    # 3. Set Groq API Key
    # Load environment variables from a .env file
    load_dotenv()
    if "GROQ_API_KEY" not in os.environ:
        print("Warning: GROQ_API_KEY environment variable not set.")
        print("Please set it as an environment variable or in a .env file.")
        print("Example: os.environ['GROQ_API_KEY'] = 'gsk_...'")
        # Prompt for the key if not using .env
        api = input("Enter your Groq API key: ").strip()
        if not api:
            print("API key not found. Please set it as an environment variable or in a .env file.")
            return
        os.environ["GROQ_API_KEY"] = api

    # 4. Define Tools.
    @tool
    def basic_calculator(expression: str) -> str:
        """Evaluates a mathematical expression and returns the result."""
        try:
            # Safe evaluation for basic arithmetic
            # allowed_chars = set("0123456789+-*/(). ")
            # if not all(c in allowed_chars for c in expression):
            #     return "Error: Invalid characters in expression."
            
            # Using eval with restricted builtins for basic math operations
            result = eval(expression, {"__builtins__": None}, {})
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"

    @tool
    def get_weather(location: str) -> str:
        """Returns the current weather in a specified location."""
        # Mock implementation of an external weather API
        return f"The weather in {location} is currently [Bohut haalat kharab hai idhar] with clear skies."

    # 5. Register Tools.
    tools = [basic_calculator, get_weather]

    # 6. Initialize LLM.
    try:
        # We use ChatGroq model which supports tool calling natively
        # We specify llama-3.3-70b-versatile which is a very fast open-source model available on Groq
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    # 7. Create Agent.
    # Define the prompt that guides the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided tools to answer queries effectively."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the agent that will bind the LLM, tools, and prompt
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 8. Execute agent.
    # AgentExecutor is the runtime that actually executes the agent and calls tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 9. Perform interactive chat.
    print("=========================================================")
    print("Intelligent Agent Online (Powered by Groq).")
    print("Available capabilities: Basic Math Calculator, Weather Check.")
    print("Type 'exit', 'quit', or 'q' to stop the chat.")
    print("=========================================================")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting interactive chat...")
                break
            
            if not user_input.strip():
                continue
                
            # Invoke the agent executor
            response = agent_executor.invoke({"input": user_input})
            print(f"\nAgent: {response['output']}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred during execution: {e}")

if __name__ == "__main__":
    main()
