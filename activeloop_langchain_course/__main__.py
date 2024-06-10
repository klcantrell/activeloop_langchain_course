from dotenv import load_dotenv

# from zero_to_hero import (
#     llm_chain,
#     conversation_chain,
#     first_deeplake_documents,
#     retrievalqa_chain,
#     websearch_tool,
#     websearch_summary_tools,
# )

# from intro_llm import (
#     track_token_usage,
#     few_shot,
#     question_answering_prompts,
#     summarization,
#     translation,
#     intro_to_tokens,
#     prompt_chain,
#     summarization_chain,
#     question_answer_chain,
#     conversation,
# )

# from prompting import (
#     intro as prompting_intro,
#     few_shot as prompting_few_shot,
#     chaining_prompts,
#     using_selectors as prompting_using_selectors,
#     pydantic_output_parser as prompting_pydantic_output_parser,
#     comma_separated_output_parser as prompting_comma_separated_output_parser,
#     output_fixing_parser as prompting_output_fixing_parser,
#     retry_output_parser as prompting_retry_output_parser,
#     basic_knowledge_graphs as prompting_basic_knowledge_graphs,
# )

# from indexes import (
#     intro as indexes_intro,
#     loaders as indexes_loaders,
#     text_splitters as indexes_text_splitters,
#     basic_similarity_search as indexes_basic_similarity_search,
#     hugging_face_embeddings as indexes,
#     cohere_embeddings as indexes,
#     qa_chatbot_selenium_loader as indexes_qa_chatbot_selenium_loader,
# )

# from chains import (
#     basic_llm_chain as chains_basic_llm_chain,
#     basic_parsers as chains_basic_parsers,
#     basic_memory as chains_basic_memory,
#     basic_sequential_chain as chains_basic_sequential_chain,
#     debugging_chains as chains_debugging_chains,
#     custom_chains as chains_custom_chains,
# )

# from newspaper_summarizier import (
#     basic as newspaper_summarizer_basic,
#     with_fewshot as newspaper_summarizier_with_fewshot,
#     with_parsing as newspaper_summarizer_with_parsing,
# )

# from youtube_summarizer import (
#     without_vectordb as youtube_summarizer_without_vectordb,
#     with_vectordb as youtube_summarizer_with_vectordb,
# )

# import voice_assistant

# import codebase_assistant

# from critique_chains import (
#     basic as critique_chains_basic,
#     docs_helper_example as critique_chains_docs_helper_example,
# )

# from memory import (
#     basic_memory as memory_basic,
# )

# from tools import (
#     google_search as tools_google_search,
#     requests as tools_requests,
#     python_repl as tools_python_repl,
#     wikipedia as tools_wikipedia,
#     multiple_tools as tools_multiple,
#     writing_assistant as tools_writing_assistant,
#     websearch_chatbot as tools_websearch_chatbot,
#     custom_tool as tools_custom_tool,
# )

from agents import basic_question_answering as agents_basic_question_answering

# import codebase_assistant_ui


load_dotenv()

# zero_to_hero
# llm_chain.run()
# conversation_chain.run()
# first_deeplake_documents.run()
# retrievalqa_chain.run()
# websearch_tool.run()
# websearch_summary_tools.run()

# intro_llm
# track_token_usage.run()
# few_shot.run()
# question_answering_prompts.run()
# summarization.run()
# translation.run()
# intro_to_tokens.run()
# prompt_chain.run()
# summarization_chain.run()
# question_answer_chain.run()
# conversation.run()

# prompting
# prompting_intro.run()
# chaining_prompts.run()
# prompting_using_selectors.run()
# prompting_pydantic_output_parser.run()
# prompting_comma_separated_output_parser.run()
# prompting_output_fixing_parser.run()
# prompting_retry_output_parser.run()

# newspaper summarizier projects
# newspaper_summarizer_basic.run()
# newspaper_summarizier_with_fewshot.run()
# newspaper_summarizer_with_parsing.run()
# prompting_basic_knowledge_graphs.run()


# indexes
# indexes_intro.run()
# indexes_loaders.run()
# indexes_text_splitters.run()
# indexes_basic_similarity_search.run()
# indexes.run()
# indexes_cohere_embeddings.run()
# indexes_qa_chatbot_selenium_loader.run()

# chains
# chains_basic_llm_chain.run()
# chains_basic_parsers.run()
# chains_basic_memory.run()
# chains_basic_sequential_chain.run()
# chains_debugging_chains.run()
# chains_custom_chains.run()

# youtube summarizer
# youtube_summarizer_without_vectordb.run()
# youtube_summarizer_with_vectordb.run()

# voice assistant
# voice_assistant.run() # with streamlit, command is streamlit run [path-to-file]

# codebase_assist
# codebase_assistant.run()

# critique chains
# critique_chains_basic.run()
# critique_chains_docs_helper_example.run()

# memory
# memory_basic.run()

# codebase_assist_ui
# codebase_assistant_ui.run()

# tools
# tools_google_search.run()
# tools_requests.run()
# tools_python_repl.run()
# tools_wikipedia.run()
# tools_multiple.run()
# tools_writing_assistant.run()
# tools_websearch_chatbot.run()
# tools_custom_tool.run()

# agents
agents_basic_question_answering.run()
