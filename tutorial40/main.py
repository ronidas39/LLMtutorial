from langchain_openai import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate,ChatPromptTemplate,PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain.chains import LLMChain

llm=ChatOpenAI(model="gpt-4",temperature=0.0)

tagline=ResponseSchema(
    name="tagline",description="generated tagline for the input company description"
)
rating=ResponseSchema(
    name="rating",description="this is a whole number ,generated rating between 1-100 for the input company description"
)
rs=[tagline,rating]
output_parser=StructuredOutputParser.from_response_schemas(rs)
format_instruction=output_parser.get_format_instructions()
ts="""
you are master at suggesting unique tagline for a company based on a input description.
take the company description below delimited by triple backticks and use it to create the unique tagline.
input description: ```{input}```
the based on the input you should create a tagline and popularity score for the generates tagline between 1-100 basaed on your knowledge and analysis.
{format_instruction}
"""
prompt=ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(ts)
    ],
    input_variables=["input"],
    partial_variables={
    "format_instruction":format_instruction},
    output_parser=output_parser
)
chain=LLMChain(llm=llm,prompt=prompt)
response=chain.predict_and_parse(input="this is company makes cool and trendy watches for indian youth ")
print(response["tagline"])