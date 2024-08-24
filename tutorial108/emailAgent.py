from langchain.agents.agent_toolkits import GmailToolkit
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType,initialize_agent

toolkit=GmailToolkit()
llm=ChatOpenAI(model="gpt-4o")
agent=initialize_agent(tools=toolkit.get_tools(),llm=llm,verbose=True,max_iterations=1000,max_execution_time=1600,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
#agent.handle_parsing_errors=True
def runAgent(input):
    try:
        response=agent.run(input)
        return response
    except Exception as e:
        return response