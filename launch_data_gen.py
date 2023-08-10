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
from langchain.text_splitter import CharacterTextSplitter


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


def chunk_text(text: str, chunk_size: int = 4096, encoding: int = 4) -> list[str]:
    chunks = []
    current_chunk = []

    for token in encoding:
        current_chunk.append(token)
        if sum(len(t) for t in current_chunk) >= chunk_size:
            chunks.append("".join(current_chunk))
            current_chunk = []

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks


def launch_data_gen(
    url_docs,
    documents_embeds,
    model_name="gpt-4",
    logger=None,
    documents_for_summary=None,
    spinLogger=None,
    token_limit=4096,
    api_key=None,
):
    openai.api_key = api_key

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

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
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

    generated_instructons = []

    for _, doc in tqdm(enumerate(embed_docs)):
        instruction = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": instructions_gen_prompt.format(
                        summaries=truncate(encoding, doc.page_content, 5000),
                        url_docs=url_docs,
                        prompt=seed_instructions["instruction"],
                    ),
                }
            ],
            max_tokens=700,
        )["choices"][0]["message"]["content"]
        generated_instructons.append(instruction)

    generated_instructons_text = " ".join(generated_instructons)

    logger.info("Instruction Generation ends")
    logger.info("Code Generation Begins")

    spinLogger.info("Instruction Generation ends....")
    spinLogger.info("Code Generation Begins....")

    related_docs = documents_embeds.similarity_search(generated_instructons_text, k=2)
    related_docs.extend(
        documents_embeds.similarity_search(seed_instructions["instruction"], k=2)
    )

    code_template = open("assets/prompt_input_code.txt").read() + "\n"
    code_prompt = PromptTemplate(
        input_variables=["related_docs", "instructions", "prompt"],
        template=code_template,
    )
    
    generated_code = []
    for _, instruction in tqdm(enumerate(generated_instructons)):
        code_blocks = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": code_prompt.format(
                        instructions=truncate(encoding, instruction, 5000),
                        related_docs=related_docs,
                        prompt=seed_instructions["instruction"],
                    ),
                }
            ],
            max_tokens=700,
        )["choices"][0]["message"]["content"]
        generated_code.append(code_blocks)

    code_text = " "

    print(generated_code)

    for current_block in generated_code:
        code_output = extract_code_block(current_block)
        code_text += code_output + "\n"

    logger.info("Code Generation Ends...")
    spinLogger.info("Code Generation Ends...")
    spinLogger.info("Proof Reading code...")

    output_data = {"input": seed_instructions["instruction"], "output": code_text}

    return output_data
