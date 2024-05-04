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

# from newspaper_summarizier import (
#     basic as newspaper_summarizer_basic,
# )

from prompting import (
    # intro as prompting_intro,
    # few_shot as prompting_few_shot,
    # chaining_prompts,
    # using_selectors as prompting_using_selectors,
    # pydantic_output_parser as prompting_pydantic_output_parser,
    # comma_separated_output_parser as prompting_comma_separated_output_parser,
    # output_fixing_parser as prompting_output_fixing_parser,
    retry_output_parser as prompting_retry_output_parser,
)


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

# newspaper summarizier projects
# newspaper_summarizer_basic.run()

# prompting
# prompting_intro.run()
# chaining_prompts.run()
# prompting_using_selectors.run()
# prompting_pydantic_output_parser.run()
# prompting_comma_separated_output_parser.run()
# prompting_output_fixing_parser.run()
prompting_retry_output_parser.run()
