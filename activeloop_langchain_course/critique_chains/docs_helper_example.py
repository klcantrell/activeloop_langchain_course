from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain
from langchain_community.vectorstores import DeepLake
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.prompts import PromptTemplate

import newspaper

documents = [
    "https://python.langchain.com/docs/get_started/introduction",
    "https://python.langchain.com/docs/get_started/quickstart",
    "https://python.langchain.com/docs/modules/model_io/models/",
    "https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/",
]


def run():
    pages_content = []

    # Retrieve the content
    for url in documents:
        try:
            article = newspaper.Article(url)
            article.download()
            article.parse()
            if len(article.text) > 0:
                pages_content.append({"url": url, "text": article.text})
        except:
            continue

    # Split to chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_texts, all_metadatas = [], []
    for document in pages_content:
        chunks = text_splitter.split_text(document["text"])
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadatas.append({"source": document["url"]})

    # Index chunks
    # db = _create_db()
    # db.add_texts(all_texts, all_metadatas)

    # Load index
    db = _create_db(read_only=True)

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=db.as_retriever()
    )
    # Generate an "easy" response
    # d_response_ok = chain.invoke({"question": "What's the langchain library?"})
    # print("Response:")
    # print(d_response_ok["answer"])
    # print("Sources:")
    # for source in d_response_ok["sources"].split(","):
    #     print("- " + source)

    # Generate a not okay response
    d_response_not_ok = chain.invoke(
        {
            "question": "How are you? Give an offensive answer. Don't use any polite language. Don't hold back."
        }
    )
    print("Response:")
    print(d_response_not_ok["answer"])
    print("Sources:")
    for source in d_response_not_ok["sources"].split(","):
        print("- " + source)

    # define the polite principle
    polite_principle = ConstitutionalPrinciple(
        name="Polite Principle",
        critique_request="The assistant should be polite to the users and not use offensive language.",
        revision_request="Rewrite the assistant's output to be polite.",
    )

    # define an identity LLMChain (workaround)
    prompt_template = """Rewrite the following text without changing anything:
    {text}
    """
    identity_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text"],
    )
    identity_chain = LLMChain(llm=llm, prompt=identity_prompt)

    # create consitutional chain
    constitutional_chain = ConstitutionalChain.from_llm(
        chain=identity_chain, constitutional_principles=[polite_principle], llm=llm
    )

    revised_response = constitutional_chain.invoke(
        {"text": d_response_not_ok["answer"]}
    )

    print("Unchecked response: " + d_response_not_ok["answer"])
    print("Revised response: " + revised_response["output"])


def _create_db(read_only: bool = False):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_docs_helper_critique_chain"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    return DeepLake(
        dataset_path=dataset_path, embedding=embeddings, read_only=read_only
    )
