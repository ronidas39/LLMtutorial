from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
raw_pdf_elements = partition_pdf(
    filename=r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial33\sample.pdf",
    
    # Using pdf format to find embedded image blocks
    extract_images_in_pdf=True,
    
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    # Hard max on chunks
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path="static/pdfImages/",
)
