from typing import TypedDict

from langchain_openai import OpenAI
from langchain.agents import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import newspaper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


def run():
    query = "What is the latest fast and furious movie?"

    web_search_results = _get_web_search_results(query)
    pages_content = _get_pages_content(web_search_results)
    documents = _get_document_chunks(pages_content)
    top_documents = _get_top_documents(documents, query)

    chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")

    response = chain.invoke(
        {"input_documents": top_documents, "question": query},
        return_only_outputs=True,
    )

    response_text, response_sources = response["output_text"].split("SOURCES:")
    response_text = response_text.strip()
    response_sources = response_sources.strip()

    print(f"Answer: {response_text}")
    print(f"Sources: {response_sources}")


def _get_web_search_results(query: str):
    search = GoogleSearchAPIWrapper()
    topNResults = 5

    def top_n_results(query):
        return search.results(query, topNResults)

    tool = Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=top_n_results,
    )

    results = tool.run(query)

    return [result["link"] for result in results]


PageContent = TypedDict("PageContent", {"url": str, "text": str})


def _get_pages_content(web_links: list[str]) -> list[PageContent]:
    pages_content: list[PageContent] = []

    for link in web_links:
        try:
            article = newspaper.Article(link)
            article.download()
            article.parse()

            if len(article.text) > 0:
                pages_content.append({"url": link, "text": article.text})
        except Exception as _e:
            continue

    return pages_content


def _get_document_chunks(pages_content: list[PageContent]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)

    docs: list[Document] = []
    for d in pages_content:
        chunks = text_splitter.split_text(d["text"])
        for chunk in chunks:
            new_doc = Document(page_content=chunk, metadata={"source": d["url"]})
            docs.append(new_doc)

    return docs


def _get_top_documents(documents: list[Document], query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    docs_embeddings = embeddings.embed_documents(
        [doc.page_content for doc in documents]
    )
    query_embedding = embeddings.embed_query(query)

    top_k = 3
    best_indexes = get_top_k_indices(docs_embeddings, query_embedding, top_k)
    best_k_documents = [doc for i, doc in enumerate(documents) if i in best_indexes]
    return best_k_documents


def get_top_k_indices(list_of_doc_vectors, query_vector, top_k):
    # convert the lists of vectors to numpy arrays
    list_of_doc_vectors = np.array(list_of_doc_vectors)
    query_vector = np.array(query_vector)

    # compute cosine similarities
    similarities = cosine_similarity(
        query_vector.reshape(1, -1), list_of_doc_vectors
    ).flatten()

    # sort the vectors based on cosine similarity
    sorted_indices = np.argsort(similarities)[::-1]

    # retrieve the top K indices from the sorted list
    top_k_indices = sorted_indices[:top_k]

    return top_k_indices
