from langchain_community.embeddings import HuggingFaceHubEmbeddings

embeddings = HuggingFaceHubEmbeddings(model="https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5",
                                      huggingfacehub_api_token="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")

text = "What is deep learning?"

query_result = embeddings.embed_query(text)
print(query_result)