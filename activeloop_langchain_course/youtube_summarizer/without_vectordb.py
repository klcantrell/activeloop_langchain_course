from langchain_openai import OpenAI
from langchain_core.language_models import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import textwrap
import whisper
import yt_dlp


def run():
    # _download_mp4_from_youtube("https://www.youtube.com/watch?v=mBjPyte2ZZo")
    # _transcribe()

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

    with open(
        "activeloop_langchain_course/youtube_summarizer/lecuninterviewtranscription.txt"
    ) as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    texts = text_splitter.split_text(text)
    # just using the first four chunks for demo purposes
    docs = [Document(page_content=t) for t in texts[:4]]

    _summarize_default_mapreduce(llm, docs)
    _summarize_custom(llm, docs)
    _summarize_default_refine(llm, docs)


def _summarize_default_refine(llm: BaseLLM, docs: list[Document]):
    chain = load_summarize_chain(llm, chain_type="refine")
    output_summary = chain.invoke({"input_documents": docs})
    wrapped_text = textwrap.fill(output_summary["output_text"], width=100)
    print(wrapped_text)


def _summarize_default_mapreduce(llm: BaseLLM, docs: list[Document]):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    output_summary = chain.invoke({"input_documents": docs})
    wrapped_text = textwrap.fill(output_summary["output_text"], width=100)
    print(wrapped_text)


def _summarize_custom(llm: BaseLLM, docs: list[Document]):
    prompt_template = """Write a concise bullet point summary of the following:


    {text}


    CONSCISE SUMMARY IN BULLET POINTS:"""
    BULLET_POINT_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["text"]
    )
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=BULLET_POINT_PROMPT)
    output_summary = chain.invoke({"input_documents": docs})
    wrapped_text = textwrap.fill(
        output_summary["output_text"],
        width=1000,
        break_long_words=False,
        replace_whitespace=False,
    )
    print(wrapped_text)


def _transcribe():
    model = whisper.load_model("base")
    result = model.transcribe(
        "activeloop_langchain_course/youtube_summarizer/lecuninterview.mp4"
    )
    print(result["text"])
    with open(
        "activeloop_langchain_course/youtube_summarizer/lecuninterviewtranscription.txt",
        "w",
    ) as file:
        file.write(result["text"])


def _download_mp4_from_youtube(url):
    # Set the options for the download
    filename = "activeloop_langchain_course/youtube_summarizer/lecuninterview.mp4"
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl": filename,
        "quiet": True,
        "nocheckcertificate": True,  # workaround for netskope corporate proxy
    }

    # Download the video file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
