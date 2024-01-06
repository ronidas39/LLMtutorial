import streamlit as st
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
from sqlalchemy import create_engine

st.set_page_config(page_title="AI APP TO CHAT WITH SQL DB")
st.header="ASK ANYTHING ABOUT YOUR DB"
query=st.text_input("ask question here")

cs="mssql+pymssql://sa:xxxxx@localhost/test"
db_engine=create_engine(cs)
db=SQLDatabase(db_engine)

llm=ChatOpenAI(temperature=0.0,model="gpt-4")
sql_toolkit=SQLDatabaseToolkit(db=db,llm=llm)
sql_toolkit.get_tools()

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        you are a very intelligent AI assistant who is expert in identifing relevant questions from user and converting into sql queriers to generate coorect answer.
        Please use the belolw context to write the microsoft sql queries, dont use mysql queries.
        context:
        you must query against the connected database,it has total 5 tables,these are Customer,Order,OrderItem,Product,Supplier.
        Customer table has Id,FirstName,LastName,City,Country,Phone columns.It gives the customer information.
        Order table has Id,OrderDate,OrderNumber,CustomerId,TotalAmount columns.This gives order specific information.
        Product table has Id,ProductName,SupplierId,UnitPricePackage,IsDiscontinued columns.This gives information on products.
        Supplier table has Id,CompanyName,ContactName,ContactTitle,City,Country,Phone,Fax columns.This table gives information on the suppliers.
        OrderItem table has Id,OrderId,ProductId,UnitPrice,Quantity columns.It gives information on orderd products.
        As an exper you must use joins whenewver required.
        """
        ),
        ("user","{question}\ ai: ")
    ]

        )
agent=create_sql_agent(llm=llm,toolkit=sql_toolkit,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,max_execution_time=100,max_iterations=1000)

if st.button("Submit",type="primary"):
    if query is not None:
        response=agent.run(prompt.format_prompt(question=query))
        st.write(response)