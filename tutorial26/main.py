from langchain.document_loaders import JSONLoader

loader=JSONLoader(file_path="sample.json",jq_schema=".")