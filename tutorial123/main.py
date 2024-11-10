from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm=ChatOpenAI(model="gpt-4o")

prompt=ChatPromptTemplate.from_template("explain the {topic}")
chain=prompt | llm | StrOutputParser()

final_prompt=ChatPromptTemplate.from_template("give the usage of the {following}")
final_chain = {"following":chain} | final_prompt | llm | StrOutputParser()

response=final_chain.invoke({"topic":"Pythagorean Theorem"})
print(response)


from langchain_core.runnables import RunnableParallel

final_chain=(
                                RunnableParallel({"following":chain})
                               .pipe(final_prompt)
                               .pipe(llm)
                               .pipe(StrOutputParser())

)

response=final_chain.invoke({"topic":"Pythagorean Theorem"})
print(response)

final_chain=RunnableParallel({"following":chain}).pipe(final_prompt,llm,StrOutputParser())
response=final_chain.invoke({"topic":"Pythagorean Theorem"})
print(response)