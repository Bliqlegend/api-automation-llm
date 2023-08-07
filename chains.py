from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()


def truncate(encoding, prompt, max_size):
    input_ids = encoding.encode(prompt, disallowed_special="all")
    truncated_ids = input_ids[:max_size]
    return encoding.decode(truncated_ids)


def summarize_chain(summary_template, temperature, doc, encoding_gpt3):
    summary_prompt = PromptTemplate(
        input_variables=["passage"], template=summary_template
    )

    llm = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt)
    return chain.run(passage=truncate(encoding_gpt3, doc.page_content, 3100))
