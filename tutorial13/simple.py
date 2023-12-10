from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType,initialize_agent,load_tools
import os
llm=ChatOpenAI(temperature=0,model="gpt-4")
os.environ["SERPAPI_API_KEY"]="6bc26c990eab01a865f08b77ed174dbe4eb5a5b7f031b6a8ee621187228cf95e"
tools=load_tools(["serpapi","llm-math"],llm=llm)

with get_openai_callback() as cost:
    result=llm.invoke("what is blockchain")
    print(cost)