from langchain_community.embeddings import HuggingFaceEmbeddings


def run():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    documents = ["Document 1", "Document 2", "Document 3"]
    doc_embeddings = hf.embed_documents(documents)
    print(doc_embeddings)
