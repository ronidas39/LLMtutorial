from langchain_openai import OpenAI
from langchain.chains import APIChain


api_docs="""
base_url:http://localhost:8000
end point getzip/{name} uses GET requrest to give zip code of any given usa city,
here name is url parameter which is actually the name of the city
"""
llm=OpenAI(temperature=0)
chain=APIChain.from_llm_and_api_docs(
    llm,
    api_docs,
    verbose=True,
    limit_to_domains=["http://localhost:8000"]
)
response=chain.invoke("what is the zip code for Florida")
print(response)