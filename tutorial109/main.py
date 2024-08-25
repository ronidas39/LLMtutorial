from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import create_openai_functions_agent,AgentExecutor
search=TavilySearchResults()
llm=ChatOpenAI(model="gpt-4o")
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are an expert in searching internet and returning output with nice formatted paragraphs"),
        ("human","{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)
tools=[search]
agent=create_openai_functions_agent(llm=llm,tools=tools,prompt=prompt)
executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
response=executor.invoke({"input":"what is the latest news in Cricket in as per todays date 25th august , 2024"})
print(response["output"])