from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter,
)


def run():
    # _character_text_splitter()
    # _recursive_character_text_splitter()
    # _nltk_text_splitter()
    # _markdown_text_splitter()
    _token_text_splitter()


def _character_text_splitter():
    loader = PyPDFLoader("activeloop_langchain_course/indexes/react-19-beta.pdf")
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(pages)

    print(f"You have {len(texts)} documents")
    print("Preview:")
    print(texts[0].page_content)


def _recursive_character_text_splitter():
    loader = PyPDFLoader("activeloop_langchain_course/indexes/react-19-beta.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        length_function=len,
    )

    docs = text_splitter.split_documents(pages)
    for doc in docs:
        print(doc)


def _nltk_text_splitter():
    with open("activeloop_langchain_course/indexes/example.txt") as f:
        sample_text = f.read()

    text_splitter = NLTKTextSplitter(chunk_size=500)
    texts = text_splitter.split_text(sample_text)
    print(texts)


def _markdown_text_splitter():
    markdown_text = """
    # 

    # Welcome to My Blog!

    ## Introduction
    Hello everyone! My name is **John Doe** and I am a _software developer_. I specialize in Python, Java, and JavaScript.

    Here's a list of my favorite programming languages:

    1. Python
    2. JavaScript
    3. Java

    You can check out some of my projects on [GitHub](https://github.com).

    ## About this Blog
    In this blog, I will share my journey as a software developer. I'll post tutorials, my thoughts on the latest technology trends, and occasional book reviews.

    Here's a small piece of Python code to say hello:

    \''' python
    def say_hello(name):
        print(f"Hello, {name}!")

    say_hello("John")
    \'''

    Stay tuned for more updates!

    ## Contact Me
    Feel free to reach out to me on [Twitter](https://twitter.com) or send me an email at johndoe@email.com.

    """

    markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = markdown_splitter.create_documents([markdown_text])

    print(docs)


def _token_text_splitter():
    # Load a long document
    with open("activeloop_langchain_course/indexes/example.txt") as f:
        sample_text = f.read()

    # Initialize the TokenTextSplitter with desired chunk size and overlap
    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=50)

    # Split into smaller chunks
    texts = text_splitter.split_text(sample_text)
    print(texts[0])
