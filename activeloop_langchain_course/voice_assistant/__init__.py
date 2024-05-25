import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
from langchain_community.document_loaders import TextLoader
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from elevenlabs import save
from elevenlabs.client import ElevenLabs
from streamlit_chat import message

# Constants
TEMP_AUDIO_PATH = "temp_audio.wav"
TEMP_ELEVEN_PATH = "temp_eleven.mp3"
AUDIO_FORMAT = "audio/wav"

# Load environment variables from .env file and return the keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
eleven_api_key = os.environ.get("ELEVEN_API_KEY")

load_dotenv()


def run():
    # _index_docs()

    # Initialize clients
    openai_client = OpenAI(api_key=openai_api_key)
    eleven_labs_client = ElevenLabs(api_key=eleven_api_key)

    # Initialize Streamlit app with a title
    st.write("# Puter 🖥️")

    # Load embeddings and the DeepLake database
    db = _create_db()

    # Record and transcribe audio
    transcription = _record_and_transcribe_audio(openai_client)

    # Get user input from text input or audio transcription
    user_input = _get_user_input(transcription)

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    # Search the database for a response based on user input and update the session state
    if user_input:
        output = search_db(user_input, db)
        print(output["source_documents"])
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        _display_conversation(eleven_labs_client, st.session_state)


# functions for chat UI


# Transcribe audio using OpenAI Whisper API
def _transcribe_audio(audio_file_path, openai_client: OpenAI):
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
        return response.text
    except Exception as e:
        print(f"Error calling Whisper API: {str(e)}")
        return None


# Record audio using audio_recorder and transcribe using transcribe_audio
def _record_and_transcribe_audio(openai_client: OpenAI):
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)

        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = _transcribe_audio(TEMP_AUDIO_PATH, openai_client)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)

    return transcription


# Display the transcription of the audio on the app
def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.txt", "w+") as f:
            f.write(transcription)
    else:
        st.write("Error transcribing audio.")


# Get user input from Streamlit text input field
def _get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "", key="input")


# Search the database for a response based on the user's query
def search_db(user_input, db):
    print(user_input)
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["k"] = 4
    model = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa({"query": user_input})


# Display conversation history using Streamlit messages
def _display_conversation(eleven_labs_client: ElevenLabs, history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))
        # Voice using Eleven API
        voice = "Rachel"
        text = history["generated"][i]
        audio = eleven_labs_client.generate(text=text, voice=voice)
        save(audio, TEMP_ELEVEN_PATH)
        with open(TEMP_ELEVEN_PATH, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
        os.remove(TEMP_ELEVEN_PATH)


# functions for creating the index


def _create_db(read_only: bool = False):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_jarvis_assistant"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    return DeepLake(
        dataset_path=dataset_path, embedding=embeddings, read_only=read_only
    )


def _get_documentation_urls():
    # List of relative URLs for Hugging Face documentation pages, commented a lot of these because it would take too long to scrape all of them
    return [
        "/docs/huggingface_hub/guides/overview",
        "/docs/huggingface_hub/guides/download",
        "/docs/huggingface_hub/guides/upload",
        "/docs/huggingface_hub/guides/hf_file_system",
        "/docs/huggingface_hub/guides/repository",
        "/docs/huggingface_hub/guides/search",
    ]


def _construct_full_url(base_url, relative_url):
    # Construct the full URL by appending the relative URL to the base URL
    return base_url + relative_url


def _scrape_page_content(url):
    # Send a GET request to the URL and parse the HTML response using BeautifulSoup
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # Extract the desired content from the page (in this case, the body text)
    text = soup.body.text.strip()
    # Remove non-ASCII characters
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]", "", text)
    # Remove extra whitespace and newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _scrape_all_content(base_url, relative_urls, filename):
    # Loop through the list of URLs, scrape content and add it to the content list
    content = []
    for relative_url in relative_urls:
        full_url = _construct_full_url(base_url, relative_url)
        scraped_content = _scrape_page_content(full_url)
        content.append(scraped_content.rstrip("\n"))

    # Write the scraped content to a file
    with open(filename, "w", encoding="utf-8") as file:
        for item in content:
            file.write("%s\n" % item)

    return content


# Define a function to load documents from a file
def _load_docs(filename):
    # Create an empty list to hold the documents
    docs = []
    try:
        # Load the file using the TextLoader class and UTF-8 encoding
        loader = TextLoader(filename, encoding="utf-8")
        # Split the loaded file into separate documents and add them to the list of documents
        docs.extend(loader.load_and_split())
    except Exception as _e:
        # If an error occurs during loading, ignore it and return an empty list of documents
        pass
    # Return the list of documents
    return docs


def _split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def _index_docs():
    base_url = "https://huggingface.co"
    # Set the name of the file to which the scraped content will be saved
    filename = "activeloop_langchain_course/youtube_summarizer/content.txt"
    relative_urls = _get_documentation_urls()
    # Scrape all the content from the relative URLs and save it to the content file
    _content = _scrape_all_content(base_url, relative_urls, filename)
    # Load the content from the file
    docs = _load_docs(filename)
    # Split the content into individual documents
    texts = _split_docs(docs)
    # Create a DeepLake database with the given dataset path and embedding function
    db = _create_db()
    # Add the individual documents to the database
    db.add_documents(texts)
    # Clean up by deleting the content file
    os.remove(filename)


if __name__ == "__main__":
    run()
