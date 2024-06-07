from langchain_openai import OpenAI
from langchain.chains import APIChain


api_docs="""
base_url:https://api.coincap.io
end point /v2/assets/{name} uses GET requrest to give information on any crypto token,
here name is url parameter which is actually the name of the crytop token in lower case
"""
llm=OpenAI(temperature=0)
chain=APIChain.from_llm_and_api_docs(
    llm,
    api_docs,
    verbose=True,
    limit_to_domains=["https://api.coincap.io"]
)
response=chain.invoke("what is the price of ethereum")
print(response)