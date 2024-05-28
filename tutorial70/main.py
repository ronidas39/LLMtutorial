from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
llm=ChatGroq(model="llama3-8b-8192")
system="you are an inteeligent assistant"
human="{text}"
prompt=ChatPromptTemplate.from_messages([
    ("system",system),
    ("human",human)
]
)
chain=prompt | llm
response=chain.invoke({"text":"write all formulas for differentiation"})
print(response.content)