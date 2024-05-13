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

from indexes import (
    # intro as indexes_intro,
    # loaders as indexes_loaders,
    # text_splitters as indexes_text_splitters,
    # basic_similarity_search as indexes_basic_similarity_search,
    # hugging_face_embeddings as indexes,
    # cohere_embeddings as indexes,
    qa_chatbot_selenium_loader as indexes_qa_chatbot_selenium_loader,
)

# from newspaper_summarizier import (
#     basic as newspaper_summarizer_basic,
#     with_fewshot as newspaper_summarizier_with_fewshot,
#     with_parsing as newspaper_summarizer_with_parsing,
# )

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
indexes_qa_chatbot_selenium_loader.run()
