from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.dataset import EvaluationDataset
import deepeval
import os
from dotenv import load_dotenv
load_dotenv()
dataset=EvaluationDataset()
dataset.generate_goldens_from_docs(
    max_goldens_per_context=2,
    document_paths=["book.txt"],
    context_construction_config=ContextConstructionConfig(max_contexts_per_document=25,max_context_length=5,chunk_size=600)
)
#deepeval.login_with_confident_api_key(api_key=os.getenv("deepeval_api_key"))
dataset.push("rag dataset")