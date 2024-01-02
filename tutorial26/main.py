from langchain.document_loaders import JSONLoader
loader=JSONLoader(file_path="sample.json",jq_schema=".",json_lines=True,text_content=False)
docs=loader.load()
for doc in docs:
    print(doc.page_content)