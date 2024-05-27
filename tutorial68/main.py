from langchain_experimental.data_anonymizer import PresidioAnonymizer
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
anonymizer=PresidioAnonymizer()

text="""
Mark George recently lost his car from the city mall parking in Newyork,the incident happend on 3 rd may 2024
his car details are as below:
car model audi q76 , car number is mg6567
"""
template="""
Write an official email with this text to address city insurance company
{anonymized_text}
"""
prompt=PromptTemplate.from_template(template)
llm=ChatOpenAI(model="gpt-4o")
chain={"anonymized_text":anonymizer.anonymize} | prompt | llm
response=chain.invoke(text)
print(response.content)