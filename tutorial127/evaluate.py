from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.dataset import EvaluationDataset
import deepeval
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric
from deepeval.test_case import LLMTestCase
load_dotenv()


llm=ChatOpenAI(model="gpt-4o")
embeddings=OpenAIEmbeddings()
arm=AnswerRelevancyMetric()
fm=FaithfulnessMetric()
template="""
you are an intelligent ai assitant who can answer any {query} asked by user
based on the given {context} only,
if you do noyt found and relevant information on the given context then respond only "I DONT KNOW"
"""
prompt=ChatPromptTemplate.from_template(template)
vs=Chroma(persist_directory="./index",embedding_function=embeddings)
retriever=vs.as_retriever(search_kwargs={"k":5})
qa_chain=RetrievalQA.from_chain_type(llm=llm,retriever=retriever)

dataset=EvaluationDataset()
dataset.pull("rag dataset",auto_convert_goldens_to_test_cases=False)
test_casaes=[]
for golden in dataset.goldens[:20]:
    query=golden.input
    expected_output=golden.expected_output
    actual_output=qa_chain.invoke(query)["result"]
    actual_context=[doc.page_content for doc in retriever.invoke(query)]
    test_case=LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=actual_context,
        expected_output=expected_output
    )
    test_casaes.append(test_case)
  
deepeval.evaluate(
    test_casaes,
    metrics=[arm,fm]
)
