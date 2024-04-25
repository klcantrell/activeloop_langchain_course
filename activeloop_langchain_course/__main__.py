from dotenv import load_dotenv

from zero_to_hero import (
    # llm_chain,
    # conversation_chain,
    # first_deeplake_documents,
    # retrievalqa_chain,
    # websearch_tool,
    websearch_summary_tools,
)

load_dotenv("..")

# llm_chain.run()
# conversation_chain.run()
# first_deeplake_documents.run()
# retrievalqa_chain.run()
# websearch_tool.run()
websearch_summary_tools.run()
