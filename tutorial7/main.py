from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
from sqlalchemy import create_engine

cs="mssql+pymssql://sa:Rambo1234@localhost/test"
db_engine=create_engine(cs)
db=SQLDatabase(db_engine)


llm=ChatOpenAI(temperature=0.0,model="gpt-4")
sql_toolkit=SQLDatabaseToolkit(db=db,llm=llm)
sql_toolkit.get_tools()

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        you are a very intelligent AI assitasnt who is expert in identifying relevant questions from user and converting into sql queriesa to generate correcrt answer.
        Please use the below context to write the microsoft sql queries , dont use mysql queries.
       context:
       you must query against the connected database, it has total 5 tables , these are Customer,Order,Product,Supplier,OrderItem.
       Customer table has Id,FirstName,LastName,City,Country,Phone columns.It gives the customer information.
       Order table has Id,OrderDate,OrderNumber,CustomerId,TotalAmount columns.It gives the order specific information.
       Product table has Id,ProductName,SupplierId,UnitPrice,Package,IsDiscontinued columns.It gives information about products.
       Supplier table has Id,CompanyName,ContactName,ContactTitle,City,Country,Phone,Fax columns.This table gives information on the supplier.
       OrderItem table has Id,OrderId,ProductId,UnitPrice,Quantity columns.It gives information of ordered products.
       As an expert you must use joins whenever required.
        """
        ),
        ("user","{question}\ ai: ")
    ]
)
agent=create_sql_agent(llm=llm,toolkit=sql_toolkit,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,max_execution_time=100,max_iterations=1000)
agent.run(prompt.format_prompt(question="write down the supplier who has maxmium number of products"))