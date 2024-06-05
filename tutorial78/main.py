from langchain.chains.api.base import APIChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

api_docs = """

BASE_URL: http://localhost:8000/

API Documentation:

The API endpoint /get/zip_code/{city} Used to find informatin about the zip code for the given us city. All URL parameters are listed below:
    - city: Name of city - Ex: Camuy, Maricao
    
The API endpoint /get/state_name/{id}} Uesd to find information about the state name for the given state code. All URL parameters are listed below:
    - id: 2 letter usa state code. Example: PR,VI
    
"""

chain = APIChain.from_llm_and_api_docs(llm, api_docs=api_docs, verbose=True,limit_to_domains=None)

response=chain.run('Can you tell me zip code for about Frederiksted?')
print()
