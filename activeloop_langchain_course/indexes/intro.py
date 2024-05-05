from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def run():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    db = create_db(read_only=True)

    # load docs into db
    # save_docs(db)

    # create retriever from db
    retriever = db.as_retriever()

    # create a retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    query = "How Google plans to challenge OpenAI?"

    # response without compressor
    print(qa_chain.invoke(query))

    # create compressor for the retriever
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    # retrieving compressed documents
    retrieved_docs = compression_retriever.invoke(
        "How Google plans to challenge OpenAI?"
    )
    print(retrieved_docs[0].page_content)

    # re-create a retrieval chain with compression retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
    )

    # response with compressor
    print(qa_chain.invoke(query))


def create_db(read_only: bool):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_indexers_retrievers"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    return DeepLake(
        dataset_path=dataset_path, embedding=embeddings, read_only=read_only
    )


def save_docs(db: DeepLake):
    # text to write to a local file
    # taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
    text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
    Google is offering developers access to one of its most advanced AI language models: PaLM.
    The search giant is launching an API for PaLM alongside a number of AI enterprise tools
    it says will help businesses “generate text, images, code, videos, audio, and more from
    simple natural language prompts.”

    PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
    Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
    PaLM is a flexible system that can potentially carry out all sorts of text generation and
    editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
    example, or you could use it for tasks like summarizing text or even writing code.
    (It’s similar to features Google also announced today for its Workspace apps like Google
    Docs and Gmail.)
    """

    # write text to local file
    with open("my_file.txt", "w") as file:
        file.write(text)

    # use TextLoader to load text from local file
    loader = TextLoader("my_file.txt")
    docs_from_file = loader.load()

    # create a text splitter
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # split documents into chunks
    docs = text_splitter.split_documents(docs_from_file)

    # add documents to our Deep Lake dataset
    db.add_documents(docs)
