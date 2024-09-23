from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from genrate_video import generateVideo

llm=ChatOpenAI(model="gpt-4o")
template="you are an intelligent & creative thinker who can express his imagination about {input} in one line with minimum words"
prompt=PromptTemplate.from_template(template)
chain=prompt|llm
response=chain.invoke({"input":"snowy night of Chicago"})
print(response.content)
generateVideo(response.content)