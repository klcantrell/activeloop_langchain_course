from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_text_splitters import RecursiveCharacterTextSplitter


def run():
    # instantiate the LLM and embeddings models
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # create our documents
    texts = [
        "Napoleon Bonaparte was born in 15 August 1769",
        "Louis XIV was born in 5 September 1638",
        "Lady Gaga was born in 28 March 1986",
        "Michael Jeffrey Jordan was born in 17 February 1963",
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(texts)

    # create Deep Lake dataset
    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

    # add documents to our Deep Lake dataset
    db.add_documents(docs)
