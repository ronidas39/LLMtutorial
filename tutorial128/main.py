from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableParallel,RunnableBranch
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from pydantic import BaseModel,Field

llm=ChatOpenAI(model="gpt-4o")
strparser=StrOutputParser()
def getDessert(input):
    type=input["type"]
    template="""
    you are an multi cuisine dessert specialist
    you have suggest a dessert name based of user dessert {type}
    reply should be only dessert name nothing else
    """
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|llm|strparser
    return chain.invoke({"type":type})

def getRecipe(menu):
    items=menu["menu"]
    template="""
    you are an multi cuisine specialist chef 
    you have to generate authentic and healthly recipe for the {item}
    reply should be only recipe nothing else
    """
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|llm|strparser
    receipes=''
    for item in items:
        recipe=chain.invoke({"item":item})
        receipes=receipes+recipe+'\n\n'
        return receipes
    
def getSummary(text):
    template="""you are an expert summary writer , provide summary for {text}"""
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|llm|strparser
    return chain.invoke({"text":text})


class cuisine(BaseModel):
    name:str=Field(description="name of dish which user wants to have")
    type:str=Field(description="cuisine type like Indian or spanish etc")
parser=JsonOutputParser(pydantic_object=cuisine)
template="""
You are an intelligent AI assistant acting as a smart, 
friendly waiter in a global cuisine restaurant.
Your job is to read the user's sentence about what they feel like eating, and return:
The cuisine type (like Indian,Italian etc) , use your knowledge to identify it
The dish name or main food-related phrase mentioned in their input
{query}
{format_instructions}
"""
prompt=PromptTemplate(
    template=template,
    input_variables=["query"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)
master_chain=(
            {"query":RunnablePassthrough()}
            |prompt
            |llm
            |parser
            |RunnableParallel({"mainDish":lambda x:x ["name"]}|{"dessert":RunnablePassthrough()|RunnableLambda(getDessert)})
            |{"menu":lambda x :list(x.values())}
            |RunnableLambda(getRecipe)
            |RunnableBranch(
                    (lambda x:len(x.split())>150,RunnableLambda(getSummary)),
                    (RunnablePassthrough())
            )
            )
response=master_chain.invoke({"i want to ear lasagne"})
print(response)
master_chain.get_graph().draw_mermaid_png(output_file_path="chain.png")