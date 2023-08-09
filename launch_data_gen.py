import json
import tiktoken
from tqdm import tqdm
import openai
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import pickle
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import re
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def truncate(encoding, prompt, max_size):
    input_ids = encoding.encode(prompt, disallowed_special="all")
    truncated_ids = input_ids[:max_size]
    return encoding.decode(truncated_ids)


def extract_code_block(text):
    pattern = r"```(?:python)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n" + matches[0].strip() + "\n"
    else:
        return None


def launch_data_gen(
    url_docs,
    documents_embeds,
    model_name="gpt-4",
    logger=None,
    documents_for_summary=None,
    spinLogger=None,
):
    seed_instructions_path = "assets/seed_instructions.json"
    logger.info("Summarization of embeddings begins")
    spinLogger.info("Summarization of embeddings begins....")

    encoding = tiktoken.encoding_for_model(model_name)

    with open(seed_instructions_path, "r") as f:
        seed_instructions = json.load(f)

    summaries = []
    embed_docs = []
    summary_prompt = open("assets/prompt_summary.txt").read() + "\n"
    for _, doc in tqdm(enumerate(documents_for_summary)):
        summary = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": summary_prompt.format(
                        passage=truncate(encoding, doc.page_content, 3100)
                    ),
                }
            ],
            max_tokens=700,
        )["choices"][0]["message"]["content"]
        summaries.append(summary)
        embed_docs.append(Document(page_content=summary))

    with open("assets/vectorstore_summary.pkl", "rb") as f:
        summary_embeddings = pickle.load(f)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(embed_docs, embeddings)

    logger.info("Summary Vectorstore is storing in assets/vectorstore_summary.pkl")
    with open("assets/vectorstore_summary.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    logger.info("Summarizing mode ends")
    logger.info("Instruction Generation begins")

    spinLogger.info("Summarizing mode ends....")
    spinLogger.info("Instruction Generation begins...")

    prompt_template = open("assets/prompt_instruction_gen.txt").read() + "\n"
    instructions_gen_prompt = PromptTemplate(
        input_variables=["url_docs", "summaries", "prompt"],
        template=prompt_template,
    )

    llm = ChatOpenAI(temperature=0.9, model_name=model_name)

    chain = LLMChain(llm=llm, prompt=instructions_gen_prompt)
    generated_instructons = chain.run(
        url_docs=url_docs,
        summaries=summary_embeddings,
        prompt=seed_instructions["instruction"],
    )

    print(generated_instructons)

    logger.info("Instruction Generation ends")
    logger.info("Code Generation Begins")

    spinLogger.info("Instruction Generation ends....")
    spinLogger.info("Code Generation Begins....")

    related_docs = documents_embeds.similarity_search(generated_instructons, k=2)
    related_docs.extend(
        documents_embeds.similarity_search(seed_instructions["instruction"], k=2)
    )

    code_template = open("assets/prompt_input_code.txt").read() + "\n"
    code_prompt = PromptTemplate(
        input_variables=["url_docs", "instructions", "related_docs", "prompt"],
        template=code_template,
    )

    llm = ChatOpenAI(temperature=0.9, model_name=model_name)

    chain = LLMChain(llm=llm, prompt=code_prompt)
    generated_code = chain.run(
        url_docs=url_docs,
        prompt=seed_instructions["instruction"],
        instructions=generated_instructons,
        related_docs=related_docs,
    )

    logger.info("Code Generation Ends...")
    spinLogger.info("Code Generation Ends...")
    spinLogger.info("Proof Reading code...")

    output_data = {
        "input": seed_instructions["instruction"],
        "output": extract_code_block(generated_code),
    }

    return output_data
