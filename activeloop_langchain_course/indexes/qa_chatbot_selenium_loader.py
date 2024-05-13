from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.url_selenium import SeleniumURLLoader
from langchain.prompts import PromptTemplate


def run():
    # _load_data()

    db = create_db(read_only=True)

    # user question
    query = "how to check disk usage in linux?"

    # retrieve relevant chunks
    docs = db.similarity_search(query)
    retrieved_chunks = [doc.page_content for doc in docs]

    # let's write a prompt for a customer support chatbot that
    # answer questions using information extracted from our db
    template = """You are an exceptional customer support chatbot that gently answer questions.

    You know the following context information.

    {chunks_formatted}

    Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

    Question: {query}

    Answer:"""

    prompt = PromptTemplate(
        input_variables=["chunks_formatted", "query"],
        template=template,
    )

    # format the prompt
    chunks_formatted = "\n\n".join(retrieved_chunks)
    prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

    # generate answer
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    answer = llm.invoke(prompt_formatted)
    print(answer)


def _load_data():
    # we'll use information from the following articles
    urls = [
        "https://beebom.com/what-is-nft-explained/",
        "https://beebom.com/how-delete-spotify-account/",
        "https://beebom.com/how-download-gif-twitter/",
        "https://beebom.com/how-use-chatgpt-linux-terminal/",
        "https://beebom.com/how-delete-spotify-account/",
        "https://beebom.com/how-save-instagram-story-with-music/",
        "https://beebom.com/how-install-pip-windows/",
        "https://beebom.com/how-check-disk-usage-linux/",
    ]

    # use the selenium scraper to load the documents
    loader = SeleniumURLLoader(urls=urls)
    docs_not_splitted = loader.load()

    # we split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(docs_not_splitted)

    db = create_db()

    # add documents to our Deep Lake dataset
    db.add_documents(docs)


def create_db(read_only: bool):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_selenium_loader_qa_chatbot"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    return DeepLake(
        dataset_path=dataset_path, embedding=embeddings, read_only=read_only
    )
