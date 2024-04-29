from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from newspaper import Article


def run():
    article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"
    article = Article(article_url)
    article.download()
    article.parse()

    # _basic_summary(article)
    _bulleted_list(article)


def _bulleted_list(article: Article):
    # prepare template for prompt
    template = """You are an advanced AI assistant that summarizes online articles into bulleted lists.

    Here's the article you need to summarize.

    ==================
    Title: {article_title}

    {article_text}
    ==================

    Now, provide a summarized version of the article in a bulleted list format.
    """

    # format prompt
    prompt = template.format(article_title=article.title, article_text=article.text)

    # load the model
    chat = ChatOpenAI(model="gpt-4", temperature=0)

    # generate summary
    summary = chat([HumanMessage(content=prompt)])
    print(summary.content)


def _basic_summary(article: Article):
    # prepare template for prompt
    template = """You are a very good assistant that summarizes online articles.

    Here's the article you want to summarize.

    ==================
    Title: {article_title}

    {article_text}
    ==================

    Write a summary of the previous article.
    """

    prompt = template.format(article_title=article.title, article_text=article.text)

    messages = [HumanMessage(content=prompt)]

    # load the model
    chat = ChatOpenAI(model="gpt-4", temperature=0)

    # generate summary
    summary = chat.invoke(messages)
    print(summary.content)
