"""
Experiment No. : 6
Title: Implementation of the ReAct (Reasoning and Acting) Loop for Autonomous Decision-Making Agents.
Objectives:
1. Understand the ReAct framework and its components (Reasoning + Acting)
2. Implement a simple autonomous agent capable of interacting with a simulated environment.
"""

# 1. Install required Python libraries.
# Run this command in your terminal before executing:
# pip install langchain-groq python-dotenv

import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 2. Define the Environment.
class SimulatedEnvironment:
    """A simulated room environment that the agent needs to escape from."""
    def __init__(self):
        self.has_key = False
        self.door_locked = True
        self.escaped = False

    def reset(self):
        self.has_key = False
        self.door_locked = True
        self.escaped = False
        return "You are in a locked room. You see a key on the table and a closed door. Your goal is to escape."

    def step(self, action: str) -> str:
        """Processes an action and returns an observation."""
        action = action.lower().strip()
        if action == "take key":
            if self.has_key:
                return "You already have the key."
            self.has_key = True
            return "You picked up the key. Now you hold a key."
        elif action == "unlock door":
            if not self.has_key:
                return "You try to unlock the door, but you don't have a key!"
            if not self.door_locked:
                return "The door is already unlocked."
            self.door_locked = False
            return "You used the key to unlock the door. It is now unlocked."
        elif action == "open door":
            if self.door_locked:
                return "You try to open the door, but it is locked. You must unlock it first."
            self.escaped = True
            return "You pushed the door open and stepped outside. You escaped!"
        elif action == "look":
            return f"Looking around: You have the key: {self.has_key}. The door is locked: {self.door_locked}."
        else:
            return f"Invalid action '{action}'. Valid actions are: look, take key, unlock door, open door."


# 3. Implement the ReAct Agent
class ReActAgent:
    def __init__(self, llm):
        self.llm = llm
        self.memory = ""

    def reason_and_act(self, observation: str):
        # 5. Do GPT-based reasoning.
        prompt = f"""
        You are an autonomous agent trying to escape a locked room.
        Use the ReAct (Reasoning and Acting) framework to make decisions.
        
        Available Actions:
        - look
        - take key
        - unlock door
        - open door
        
        Previous experiences (Memory):
        {self.memory}
        
        Current environment observation: {observation}
        
        Based on the observation and your memory, provide your output strictly formatted exactly as below:
        Thought: [Your step-by-step reasoning on what to do next to achieve the goal]
        Action: [Exactly ONE of the available actions, nothing else]
        """
        
        # Invoke the LLM to get the reasoning (Thought) and action (Action)
        response = self.llm.invoke(prompt).content
        
        # Parse the Thought and Action from the LLM's response
        thought_match = re.search(r"Thought:\s*(.*)", response, re.IGNORECASE)
        action_match = re.search(r"Action:\s*(.*)", response, re.IGNORECASE)
        
        thought = thought_match.group(1).strip() if thought_match else "No explicit thought process."
        action = action_match.group(1).strip() if action_match else "look" # fallback action
                
        # Register into memory for the next loop
        self.memory += f"\nObservation: {observation}\nThought: {thought}\nAction: {action}\n"
        
        return thought, action

# 4. Run the Agent.
def main():
    # Setup LLM based on user's Groq environment configuration
    load_dotenv()
    if "GROQ_API_KEY" not in os.environ:
        api = input("Enter your Groq API key: ").strip()
        if not api:
            print("No API key provided.")
            return
        os.environ["GROQ_API_KEY"] = api

    try:
        # We reuse the Groq model we set up previously
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    # Initialize environment and agent
    env = SimulatedEnvironment()
    agent = ReActAgent(llm)

    print("===================================================================")
    print("               Initiating ReAct Agent Simulation                   ")
    print("===================================================================")
    
    observation = env.reset()
    print(f"Initial Observation: {observation}\n")

    max_steps = 10
    step = 1

    # The continuous Reason & Act Loop
    while step <= max_steps and not env.escaped:
        print(f"--- Step {step} ---")
        
        # Agent reasons and provides an action
        thought, action = agent.reason_and_act(observation)
        print(f"🤖 Thought: {thought}")
        print(f"⚡ Action:  {action}")
        
        # Execute the action inside the simulated environment
        observation = env.step(action)
        print(f"🌍 Observation: {observation}\n")
        
        step += 1

    print("===================================================================")
    if env.escaped:
        print("🎉 SUCCESS! The ReAct agent reasoned correctly and achieved the goal!")
    else:
        print("❌ FAILED! The ReAct agent did not reach the goal within the step limit.")
    print("===================================================================")

if __name__ == "__main__":
    main()
