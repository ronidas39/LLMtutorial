from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate


llm=HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=6096,
    huggingfacehub_api_token="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"
)
pt="""
you are intelligent and funny comedian who crack one liner jokes with any laguage given asked by the user
to generate the jokes use {language} menionted by user
"""
prompt=PromptTemplate.from_template(pt)
msg=prompt.format(language="Bengali")
x=llm.invoke(msg)
print(x)



