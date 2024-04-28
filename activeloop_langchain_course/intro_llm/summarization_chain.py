from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def run():
    # Initialize language model
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

    # Load the summarization chain
    summarize_chain = load_summarize_chain(llm)

    # Load the document using PyPDFLoader
    document_loader = PyPDFLoader(
        file_path="activeloop_langchain_course/intro_llm/react-19-beta.pdf"
    )
    document = document_loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    )

    # Summarize the document
    summary = summarize_chain.invoke([document[0]])
    print(summary["output_text"])
