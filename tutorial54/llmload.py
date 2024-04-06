from langchain_community.llms import CTransformers

def loadllm():
    config={
        "max_new_tokens":1048,
        "temperature":0.1,
        "context_length":4096,
        "gpu_layers":32,
        "threads":-1
    }
    llm=CTransformers(model="/Users/roni/Documents/GitHub/LLMtutorial/tutorial54/mistral-7b-instruct-v0.1.Q5_K_S.gguf",
                      model_type="mistral",
                      config=config
                      )
    return llm