from langchain_community.agent_toolkits.gmail.toolkit import GmailToolkit
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType,initialize_agent
llm=ChatOpenAI(model="gpt-4o")
toolkit=GmailToolkit()
agent=initialize_agent(tools=toolkit.get_tools(),
                       llm=llm,
                       verbose=True,
                       max_iterations=1000,
                       max_execution_time=1600,
                       agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
def runagent(input):
    try:
        response=agent.run(input)
        return response
    except Exception as e:
        return response
    