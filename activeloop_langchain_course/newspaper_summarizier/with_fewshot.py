from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from newspaper import Article


def run():
    article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"
    article = Article(article_url)
    article.download()
    article.parse()

    # prepare template for prompt
    template = """
    As an advanced AI, you've been tasked to summarize online articles into bulleted points. Here are a few examples of how you've done this in the past:

    Example 1:
    Original Article: 'The Effects of Climate Change
    Summary:
    - Climate change is causing a rise in global temperatures.
    - This leads to melting ice caps and rising sea levels.
    - Resulting in more frequent and severe weather conditions.

    Example 2:
    Original Article: 'The Evolution of Artificial Intelligence
    Summary:
    - Artificial Intelligence (AI) has developed significantly over the past decade.
    - AI is now used in multiple fields such as healthcare, finance, and transportation.
    - The future of AI is promising but requires careful regulation.

    Now, here's the article you need to summarize:

    ==================
    Title: {article_title}

    {article_text}
    ==================

    Please provide a summarized version of the article in a bulleted list format.
    """

    prompt = template.format(article_title=article.title, article_text=article.text)

    messages = [HumanMessage(content=prompt)]

    # load the model
    chat = ChatOpenAI(model="gpt-4", temperature=0)

    # generate summary
    summary = chat.invoke(messages)
    print(summary.content)
