from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm_qsn=ChatOpenAI(model="gpt-4")
template_qsn="""
You are an experienced {department} teacher with 20 years of experience in creating question for students. 
Your task is to create one question for students

Your response should be only question nothing else
"""
prompt_qsn=PromptTemplate.from_template(template_qsn)
chain_qsn=prompt_qsn|llm_qsn
response_qsn=chain_qsn.invoke({"department":"computer science"})
print(response_qsn.content)
llm_ans=ChatOpenAI(model="gpt-4o")
template_ans="""
You are an intelligent assistant who can answer any question correctly 
Your task is to answer the {question} asked by the user

Your response should be only answer to the given question nothing else
"""
prompt_ans=PromptTemplate.from_template(template_ans)
chain_ans=prompt_ans|llm_ans
response_ans=chain_ans.invoke({"question":response_qsn.content})
print(response_ans.content)
llm=ChatOpenAI()
template="""
You are an experienced teacher with 20 years of experience assessing students' exam papers and evaluating their answers. Your task is to check the `{answer}` given by the student for a specific `{question}` and assign marks to each answer. The marking criteria are as follows:

- Minimum mark is 0, which means the answer is incorrect.
- Full marks are 5, which means the answer is completely correct.
- Assign marks accordingly based on the accuracy and completeness of the answer.

Please evaluate the answers and assign marks between 0 to 5, means marks could 0 or 1 or 2 or 3 or 4 or 5 .
Your response should be only marks nothing else
"""
prompt=PromptTemplate.from_template(template)
chain=prompt|llm
response=chain.invoke({"question":response_qsn.content,"answer":response_ans.content})
print(response.content)