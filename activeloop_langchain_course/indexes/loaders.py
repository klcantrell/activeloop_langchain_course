from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.url_selenium import SeleniumURLLoader
from langchain_community.document_loaders.pdf import PyPDFLoader


def run():
    # _text_loader()
    # _pdf_loader()
    selenium_loader()


def _text_loader():
    loader = TextLoader("activeloop_langchain_course/indexes/example.txt")
    documents = loader.load()
    print(documents)


def _pdf_loader():
    loader = PyPDFLoader("activeloop_langchain_course/indexes/react-19-beta.pdf")
    pages = loader.load_and_split()

    print(pages[0])


def selenium_loader():
    urls = [
        "https://www.youtube.com/watch?v=TFa539R09EQ&t=139s",
        "https://www.youtube.com/watch?v=6Zv6A_9urh4&t=112s",
    ]

    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()

    print(data[0])
