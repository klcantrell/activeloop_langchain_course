from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import newspaper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

title = "OpenAI CEO: AI regulation ‘is essential’"
text_all = """ Altman highlighted the potential benefits of AI technologies like ChatGPT and Dall-E 2 to help address significant challenges such as climate change and cancer, but he also stressed the need to mitigate the risks associated with increasingly powerful AI models. Altman proposed that governments consider implementing licensing and testing requirements for AI models that surpass a certain threshold of capabilities. He highlighted OpenAI’s commitment to safety and extensive testing before releasing any new systems, emphasising the company’s belief that ensuring the safety of AI is crucial. Senators Josh Hawley and Richard Blumenthal expressed their recognition of the transformative nature of AI and the need to understand its implications for elections, jobs, and security. Blumenthal played an audio introduction using an AI voice cloning software trained on his speeches, demonstrating the potential of the technology. Blumenthal raised concerns about various risks associated with AI, including deepfakes, weaponised disinformation, discrimination, harassment, and impersonation fraud. He also emphasised the potential displacement of workers in the face of a new industrial revolution driven by AI."""
text_to_change = """ Senators Josh Hawley and Richard Blumenthal expressed their recognition of the transformative nature of AI and the need to understand its implications for elections, jobs, and security. Blumenthal played an audio introduction using an AI voice cloning software trained on his speeches, demonstrating the potential of the technology."""


def run():
    # web_queries = _generate_web_queries()
    # web_search_results = _get_web_search_results(
    #     [
    #         "AI voice cloning software",
    #         "AI implications for elections",
    #         "AI implications for job displacement",
    #     ]
    # )
    pages_content = _get_pages_content(
        [
            "https://www.respeecher.com/",
            "https://www.instagram.com/senblumenthal/reel/CsZODMzJZMr/",
            "https://speechify.com/voice-cloning/",
            "https://elevenlabs.io/",
            "https://voice.ai/",
            "https://campaignlegal.org/update/how-artificial-intelligence-influences-elections-and-what-we-can-do-about-it",
            "https://www.ncsl.org/elections-and-campaigns/artificial-intelligence-ai-in-elections-and-campaigns",
            "https://www.brookings.edu/articles/the-impact-of-generative-ai-in-a-global-election-year/",
            "https://www.brennancenter.org/our-work/analysis-opinion/how-ai-puts-elections-risk-and-needed-safeguards",
            "https://www.cisa.gov/resources-tools/resources/risk-focus-generative-ai-and-2024-election-cycle",
            "https://www.forbes.com/sites/elijahclark/2023/08/18/unveiling-the-dark-side-of-artificial-intelligence-in-the-job-market/",
            "https://news.stthomas.edu/artificial-intelligence-and-its-impact-on-jobs/",
            "https://www.linkedin.com/pulse/impact-ai-job-displacement-exploring-possibilities-piyush-goyar",
            "https://www.forbes.com/sites/heatherwishartsmith/2024/02/13/not-so-fast-study-finds-ai-job-displacement-likely-substantial-yet-gradual/",
            "https://www.nexford.edu/insights/how-will-ai-affect-jobs",
        ]
    )
    documents = _get_document_chunks(pages_content)
    top_documents = _get_top_documents(documents)

    template = """You are an exceptional copywriter and content creator.

    You're reading an article with the following title:
    ----------------
    {title}
    ----------------

    You've just read the following piece of text from that article.
    ----------------
    {text_all}
    ----------------

    Inside that text, there's the following TEXT TO CONSIDER that you want to enrich with new details.
    ----------------
    {text_to_change}
    ----------------

    Searching around the web, you've found this ADDITIONAL INFORMATION from distinct articles.
    ----------------
    {doc_1}
    ----------------
    {doc_2}
    ----------------
    {doc_3}
    ----------------

    Modify the previous TEXT TO CONSIDER by enriching it with information from the previous ADDITIONAL INFORMATION.
    """

    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=template,
            input_variables=[
                "text_to_change",
                "text_all",
                "title",
                "doc_1",
                "doc_2",
                "doc_3",
            ],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)

    response = chain.invoke(
        {
            "text_to_change": text_to_change,
            "text_all": text_all,
            "title": title,
            "doc_1": top_documents[0].page_content,
            "doc_2": top_documents[1].page_content,
            "doc_3": top_documents[2].page_content,
        }
    )

    print("Text to Change: ", text_to_change)
    print("Expanded Variation:", response["text"])
    # Senators Josh Hawley and Richard Blumenthal expressed their recognition of the transformative nature of AI and the need to understand its implications for elections, jobs, and security. Blumenthal played an audio introduction using an AI voice cloning software trained on his speeches, demonstrating the potential of the technology. This demonstration comes at a critical time, as the 2024 election year will see widespread influence of AI in shaping public messages about candidates and electoral processes. The Campaign Legal Center (CLC) has been actively working to address the impact of AI on our democracy, particularly highlighting the dangers of political ads using AI technology to create deceptive content like deepfakes. These manipulations, if left unchecked, could infringe on voters' fundamental right to make informed decisions, potentially leading to scenarios where voters are misled or disenfranchised by AI-generated content. Additionally, the emergence of generative AI tools poses a significant challenge to elections, with the potential for widespread dissemination of disinformation and manipulation of voter perceptions. As AI continues to advance, there is a pressing need for policymakers to adopt measures that mitigate the risks posed by AI in campaigns and elections to ensure the integrity and fairness of the electoral process.


def _generate_web_queries():
    template = """ You are an exceptional copywriter and content creator.

    You're reading an article with the following title:
    ----------------
    {title}
    ----------------

    You've just read the following piece of text from that article.
    ----------------
    {text_all}
    ----------------

    Inside that text, there's the following TEXT TO CONSIDER that you want to enrich with new details.
    ----------------
    {text_to_change}
    ----------------

    What are some simple and high-level Google queries that you'd do to search for more info to add to that paragraph?
    Write 3 queries as a bullet point list, prepending each line with -.
    """

    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=template,
            input_variables=["text_to_change", "text_all", "title"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)

    response = chain.invoke(
        {"text_to_change": text_to_change, "text_all": text_all, "title": title}
    )

    queries = [line[2:] for line in response["text"].split("\n")]
    return queries  # ['AI voice cloning software', 'AI implications for elections', 'AI implications for job displacement']


def _get_web_search_results(queries: list[str]):
    search = GoogleSearchAPIWrapper()
    topNResults = 5

    def top_n_results(query):
        return search.results(query, topNResults)

    tool = Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=top_n_results,
    )

    all_results = []

    for query in queries:
        results = tool.run(query)
        all_results += results

    return [
        result["link"] for result in all_results
    ]  # ['https://www.respeecher.com/', 'https://www.instagram.com/senblumenthal/reel/CsZODMzJZMr/', 'https://speechify.com/voice-cloning/', 'https://elevenlabs.io/', 'https://voice.ai/', 'https://campaignlegal.org/update/how-artificial-intelligence-influences-elections-and-what-we-can-do-about-it', 'https://www.ncsl.org/elections-and-campaigns/artificial-intelligence-ai-in-elections-and-campaigns', 'https://www.brookings.edu/articles/the-impact-of-generative-ai-in-a-global-election-year/', 'https://www.brennancenter.org/our-work/analysis-opinion/how-ai-puts-elections-risk-and-needed-safeguards', 'https://www.cisa.gov/resources-tools/resources/risk-focus-generative-ai-and-2024-election-cycle', 'https://www.forbes.com/sites/elijahclark/2023/08/18/unveiling-the-dark-side-of-artificial-intelligence-in-the-job-market/', 'https://news.stthomas.edu/artificial-intelligence-and-its-impact-on-jobs/', 'https://www.linkedin.com/pulse/impact-ai-job-displacement-exploring-possibilities-piyush-goyar', 'https://www.forbes.com/sites/heatherwishartsmith/2024/02/13/not-so-fast-study-finds-ai-job-displacement-likely-substantial-yet-gradual/', 'https://www.nexford.edu/insights/how-will-ai-affect-jobs']


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


def _get_top_documents(documents: list[Document]):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    docs_embeddings = embeddings.embed_documents(
        [doc.page_content for doc in documents]
    )
    query_embedding = embeddings.embed_query(text_to_change)

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
