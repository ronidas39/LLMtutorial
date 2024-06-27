from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel,Field

class Reviews(BaseModel):
    sentiment: str =Field(description="the sentiment of the text")
    emotion: str=Field(description="what is the emotion expressed from text , is satisfied or unsatisfied")
    items:str=Field(description="the food , drinks or any the specific items mentioned")

tagging_prompt=ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.
Only extraxct the properties mentioned in the "Reviews" function

passage:
{input}
"""
)
llm=ChatOpenAI(model="gpt-4o").with_structured_output(Reviews)
tagging_chain=tagging_prompt|llm

loader=TextLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial96\review.txt")
docs=loader.load()
reviews=docs[0].page_content.split("\n")
for review in reviews:
    response=tagging_chain.invoke({"input":review})
    print(response)