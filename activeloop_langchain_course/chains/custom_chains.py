from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.callbacks.manager import CallbackManagerForChainRun

from typing import Dict, List, Optional


class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys).union(
            set(self.chain_2.input_keys)
        )
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ["concat_output"]

    def _call(
        self,
        inputs: Dict[str, str],
        _run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        output_1 = self.chain_1.invoke(inputs)
        output_2 = self.chain_2.invoke(inputs)
        return {"concat_output": output_1["text"] + output_2["text"]}


def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

    prompt_1 = PromptTemplate(
        input_variables=["word"],
        template="What is the meaning of the following word '{word}'?",
    )
    chain_1 = LLMChain(llm=llm, prompt=prompt_1)

    prompt_2 = PromptTemplate(
        input_variables=["word"],
        template="What is a word to replace the following: {word}?",
    )
    chain_2 = LLMChain(llm=llm, prompt=prompt_2)

    concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)
    concat_output = concat_chain.invoke("artificial")
    print(f"Concatenated output:\n{concat_output['concat_output']}")
