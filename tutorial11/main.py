from langchain.agents.agent_types import AgentType
from langchain.llms.openai import OpenAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent

llm=OpenAI(temperature=0.0,max_tokens=1000)
tool=PythonREPLTool()
agent=create_python_agent(llm=llm,tool=tool,verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
print(agent.run("""Given a string abcabcbb, find the length of the longest substring without repeating characters from the main string"""))