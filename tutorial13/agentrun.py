from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType,initialize_agent,load_tools
import os
llm=ChatOpenAI(temperature=0,model="gpt-4")
os.environ["SERPAPI_API_KEY"]="6bc26c990eab01a865f08b77ed174dbe4eb5a5b7f031b6a8ee621187228cf95e"
tools=load_tools(["serpapi","llm-math"],llm=llm)

agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,return_intermediate_steps=True)
with get_openai_callback() as cost:
    result=agent.run("who is the present prime minister of India,what is hios current age raise to the power of .5")
    print(f"total tokens: {cost.total_tokens}")
    print(f"prompt tokens: {cost.prompt_tokens}")
    print(f"completion tokens: {cost.completion_tokens}")
    print(f"cost is: {cost.total_cost}")