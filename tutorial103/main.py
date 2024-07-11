from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm=ChatOpenAI(model="gpt-4o")

def generateArticle(topic):
    template="""
    you are an expert article writer who can write any article on given {topic}, dont ask followup question 
    whatever is given use that and use your intelligence to write the article
    """
    prompt=PromptTemplate.from_template(template)
    chain=prompt|llm
    response=chain.invoke({"topic":topic})
    return response.content

def translateArticle(content,language):
    template="""
    you are an expert translator who can tanslate anything into any given language.
        here you have to translate {content} into {language}.
    """
    prompt=PromptTemplate.from_template(template)
    chain=prompt|llm
    response=chain.invoke({"content":content,"language":language})
    print(response.content)


translateArticle(generateArticle("semiconductor usage"),"german")