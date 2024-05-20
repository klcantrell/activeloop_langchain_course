from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import whisper
import yt_dlp

TRANSCRIPTIONS_PATH = (
    "activeloop_langchain_course/youtube_summarizer/transcriptions.txt"
)

video_urls = [
    "https://www.youtube.com/watch?v=mBjPyte2ZZo&t=78s",
    "https://www.youtube.com/watch?v=cjs7QKJNVYM",
]


def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    db = _create_db()

    # video_info = _download_mp4_from_youtube(video_urls, 1)
    # _transcribe(video_info)
    # _index_transcriptions(db)

    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["k"] = 4

    prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Summarized answer in bullet points:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )

    prediction = qa.invoke(
        "Summarize the mentions of google according to their AI program"
    )
    print(prediction["result"])


def _index_transcriptions(db: VectorStore):
    # Load the texts
    with open(TRANSCRIPTIONS_PATH) as f:
        text = f.read()

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    db.add_documents(docs)


def _create_db(read_only: bool = False):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_youtube_summarizer"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    return DeepLake(
        dataset_path=dataset_path, embedding=embeddings, read_only=read_only
    )


def _transcribe(video_info: list[tuple[str, str, str]]):
    model = whisper.load_model("base")

    results = []
    for video in video_info:
        result = model.transcribe(video[0])
        results.append(result["text"])
        print(f"Transcription for {video[0]}:\n{result['text']}\n")

    with open(TRANSCRIPTIONS_PATH, "w") as file:
        for result in results:
            file.write(result)


def _download_mp4_from_youtube(urls: list[str], job_id: int):
    # This will hold the titles and authors of each downloaded video
    video_info = []

    for i, url in enumerate(urls):
        # Set the options for the download
        file_temp = f"activeloop_langchain_course/youtube_summarizer/{job_id}_{i}.mp4"
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "outtmpl": file_temp,
            "quiet": True,
            "nocheckcertificate": True,  # workaround for netskope corporate proxy
        }

        # Download the video file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get("title", "")
            author = result.get("uploader", "")

        # Add the title and author to our list
        video_info.append((file_temp, title, author))

    return video_info
