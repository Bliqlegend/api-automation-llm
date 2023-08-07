import os
import time
import json
import random
import string
import regex as re
import pickle
import openai
import tqdm
import asyncio
import tiktoken
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from chains import summarize_chain
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def post_process_response_ins(strategy, response, **kwargs):
    """
    Post processes the given response for generating instructions based on the specified strategy.

    :param strategy: a string, represents the desired post-processing strategy for the response
    :param response: a dictionary, the response to be post-processed
    :param kwargs: keyword arguments
    :return: list of instructions
    """
    if response is None:
        return []

    if strategy == "diversifying-bing":
        num_prompt_instructions = kwargs["num_prompt_instructions"]
        raw_instructions = (
            f"{num_prompt_instructions+1}. Instruction:" + response["text"]
        )
        raw_instructions = re.split("###", raw_instructions)
    elif strategy == "summarizing-gpt-3.5-turbo-generating-gpt-4":
        num_prompt_instructions = kwargs["num_prompt_instructions"]
        if "###" in response:
            raw_instructions = re.split("###", response)
        else:
            raw_instructions = re.split("\n", response)
    else:
        raise ValueError("Unrecognised strategy provided.")

    instructions = process_raw_instructions(raw_instructions, num_prompt_instructions)
    return instructions


def process_raw_instructions(raw_instructions, num_prompt_instructions):
    """
    Processes the raw instructions for the given strategy.

    :param raw_instructions: a list of strings, instructions that are yet to be processed
    :param num_prompt_instructions: an integer, the number of prompt instructions provided
    :return: processed list of instruction dictionaries
    """
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        if idx == len(raw_instructions) - 1:
            continue

        splitted_data = re.split(
            f"{idx+num_prompt_instructions+1}\.\s+(Instruction|Question|Task):", inst
        )

        if len(splitted_data) != 3:
            inst = re.sub("(\d+)\.", "", inst)
            inst = re.sub("(Instruction|Question|Task):", "", inst)
            if is_valid_instruction(inst):
                instructions.append({"instruction": inst})
        else:
            inst = splitted_data[2].strip()
            if is_valid_instruction(inst):
                instructions.append({"instruction": inst})

    return instructions


def is_valid_instruction(instruction):
    """
    Validates if the given instruction is correct.

    :param instruction: a string, the instruction to be validated
    :return: a boolean, True if instruction is valid, otherwise False
    """
    if len(instruction.split()) <= 3 or len(instruction.split()) > 40:
        return False

    if instruction[0] in string.punctuation:
        return False

    if not instruction[0].isascii():
        return False

    return True


def post_process_response_code(response, model_name):
    """
    Post-process the given code response based on the specified model_name.

    :param response: a dictionary, the response to be post-processed
    :param model_name: a string, represents the model for which the response needs processing
    :return: a string containing the processed output
    """
    output = extract_code_output(response, model_name)

    return output


def extract_code_output(response, model_name):
    """
    Extract the code output from the given response depending on the model name.

    :param response: a dictionary, the response to be processed
    :param model_name: a string, represents the model
    :return: a string containing the code output
    """
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        output = response["choices"][0]["message"]["content"]

    return output


def extract_three_parts(output, language, code_block_start, code_block_end):
    """
    Extract the three parts of the output.

    :param output: a string containing the EdgeGPT output
    :param language: a string representing the programming language
    :param code_block_start: an integer, the starting index of the code block
    :param code_block_end: an integer, the ending index of the code block
    :return: tuple of three strings, upper part, code part, lower part of the output
    """
    upper = output[:code_block_start].replace(f"```{language}", "")
    code = output[code_block_start:code_block_end]
    lower = output[code_block_end:].replace("```", "")
    return upper, code, lower


def process_individual_lines(code_lines, part, should_add_comment=False, language=None):
    """
    Process the lines for the given part.

    :param code_lines: list, the list to append the processed lines
    :param part: a string, which part of the output it belongs to (upper, code, lower)
    :param should_add_comment: a boolean, determines if a comment should be added
    :param language: a string representing the programming language, used only for the upper part
    """
    for line in part.split("\n"):
        stripped_line = line.strip()
        if should_add_comment:
            if stripped_line.startswith("#"):
                code_lines.append(stripped_line)
            elif language is not None:
                code_lines.append(f"#{language}")
            elif stripped_line != "":
                code_lines.append("#" + stripped_line)
        else:
            code_lines.append(stripped_line)


def encode_prompt(inst_gen, url_docs, prompt_path):
    """
    Encode multiple prompt instructions into a single string.

    :param input_gen: a string, the input generator
    :param inst_gen: a string, the instruction generator
    :param url_docs: a string, url of the documentation
    :param use_scraped_docs: a boolean, if True, scraped docs will be used
    :param prompt_path: a string, the path to the prompt txt file
    :return: a string, the encoded prompt
    """
    with open(prompt_path) as file:
        prompt = file.read() + "\n"

    prompt = prompt.format(url_docs=url_docs)
    prompt += f"###\n"
    prompt += f"Instruction: {inst_gen}\n"

    return prompt


def encode_prompt_output(input_gen, inst_gen, url_docs, use_scraped_docs):
    """
    Encode multiple prompt instructions into a single string.

    :param input_gen: a string, input generator
    :param inst_gen: a string, instruction generator
    :param url_docs: a string, URL of the documentation
    :param use_scraped_docs: a boolean, if True, scraped docs will be used
    :return: a string, the encoded prompt
    """
    prompt_path = (
        "assets/prompt_input_code.txt" if use_scraped_docs else "assets/prompt_code.txt"
    )
    prompt = encode_prompt(inst_gen, url_docs, prompt_path)

    if use_scraped_docs:
        prompt += f"API References:\n{input_gen}\n"

    prompt += "Code:"
    return prompt


def encode_prompt_instruct(url, strategy, batch_size=70, **kwargs):
    """
    Encode multiple prompt instructions into a single string.

    :param url: a string, URL of the documentation or references
    :param strategy: a string, represents the desired encoding strategy
    :param batch_size: an integer, the batch size for encoding, default is 40
    :param kwargs: keyword arguments
    :return: a string, the encoded prompt
    """
    if strategy == "summarizing-gpt-3.5-turbo-generating-gpt-4":
        prompt = create_gpt_turbo_prompt(batch_size, **kwargs)
    else:
        raise ValueError("Unrecognised strategy provided.")

    return prompt


def create_gpt_turbo_prompt(batch_size, **kwargs):
    """
    Creates a GPT-3.5-turbo prompt with the given instructions.

    :param url: a string, URL of the documentation or references
    :param batch_size: an integer, the batch size
    :param kwargs: keyword arguments
    :return: a string, the GPT-3.5-turbo prompt
    """
    with open("assets/prompt_instruction_gen.txt") as file:
        prompt = file.read() + "\n"

    for idx, summary in enumerate(kwargs["summaries"]):
        prompt += f"({idx+1}) {summary}\n\n"

    batch_size += len(kwargs["prompt_instructions"])
    prompt += "###\n"
    prompt += f"List of {batch_size} tasks:\n"

    for idx, task_dict in enumerate(kwargs["prompt_instructions"]):
        instruction = task_dict["instruction"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"###\n{idx + 1}. Instruction: {instruction}\n"
    prompt += f"###\n{idx + 2}. Instruction: "
    return prompt


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def truncate(encoding, prompt, max_size):
    input_ids = encoding.encode(prompt, disallowed_special="all")
    truncated_ids = input_ids[:max_size]
    return encoding.decode(truncated_ids)


def launch_instruction_generation(
    url_docs,
    seed_instructions_path="assets/seed_instructions.json",
    strategy="summarizing-gpt-3.5-turbo-generating-gpt-4",
    num_instructions_to_generate=100,
    batch_size=70,
    temperature=0.7,
    top_p=0.7,
    logger=None,
    **kwargs,
):
    request_idx = 0
    machine_instructions = []
    request_start = time.time()

    if strategy == "summarizing-gpt-3.5-turbo-generating-gpt-4":
        """This method is a combination of summarizing and generating instructions"""
        logger.info(
            """You are using Summarizing mode with GPT-3.5 Turbo and Generating mode with GPT-4"""
        )
        logger.info("""Summarizing mode begins""")
        assert batch_size <= 80, "Batch size must be smaller than 80"
        encoding_gpt4 = tiktoken.encoding_for_model("gpt-4")
        encoding_gpt3 = tiktoken.encoding_for_model("gpt-3.5-turbo")
        with open(seed_instructions_path, "r") as f:
            seed_instructions = json.load(f)

        print(seed_instructions)

        seed_instruction_data = [
            {
                "instruction": seed_instructions["instruction"],
                "url": seed_instructions["url"],
            }
        ]

        # Get summary using gpt-3.5-turbo
        summaries = []
        embed_docs = []
        summary_prompt = open("assets/prompt_summary.txt").read() + "\n"
        for _, doc in tqdm.tqdm(enumerate(kwargs["documents_for_summary"])):
            summary = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": summary_prompt.format(
                            passage=truncate(encoding_gpt3, doc.page_content, 3100)
                        ),
                    }
                ],
                max_tokens=700,
            )["choices"][0]["message"]["content"]
            summaries.append(summary)
            embed_docs.append(Document(page_content=summary))

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(embed_docs, embeddings)

        logger.info("Summary Vectorstore is storing in assets/vectorstore_summary.pkl")
        with open("assets/vectorstore_summary.pkl", "wb") as f:
            pickle.dump(vectorstore, f)

        logger.info("Summarizing mode ends")

        logger.info("Instruction Generation begins")

        while len(machine_instructions) < num_instructions_to_generate:
            request_idx += 1
            if len(summaries) < 4:
                selected_summaries = summaries
            else:
                selected_summaries = random.sample(summaries, 4)
            max_sample_size = len(seed_instruction_data)
            num_samples = min(kwargs["num_prompt_instructions"], max_sample_size)
            prompt_instructions_gen = random.sample(seed_instruction_data, num_samples)

            kwargs_instruct = {
                "summaries": selected_summaries,
                "prompt_instructions": prompt_instructions_gen,
            }

            prompt = encode_prompt_instruct(
                url_docs, strategy, batch_size, **kwargs_instruct
            )

            max_retries = 10
            retries = 0
            while True:
                try:
                    results = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "user",
                                "content": truncate(encoding_gpt4, prompt, 6000),
                            }
                        ],
                        max_tokens=2000,
                        temperature=temperature,
                    )
                    break
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.info(f"Failed after {max_retries} attempts.")
                        raise e
                    else:
                        logger.info(
                            f"Attempt {retries} failed with exception: {e}. Retrying..."
                        )

            response = results["choices"][0]["message"]["content"]
            instruction_data = post_process_response_ins(strategy, response, **kwargs)
            for instruction_data_entry in instruction_data:
                instruction = {
                    "instruction": instruction_data_entry["instruction"],
                    "url": url_docs,
                }
                machine_instructions.append(instruction)
            request_duration = time.time() - request_start
            logger.info(f"Request {request_idx} took {request_duration:.2f}s")

    return machine_instructions


def launch_data_generation(
    url_docs,
    documents_embeds,
    output_dir,
    num_tasks_to_generate=140,
    strategy_instruct="summarizing-gpt-3.5-turbo-generating-gpt-4",
    model_name_code="gpt-4",
    num_docs_to_output=1,
    use_scraped_docs=True,
    temperature=0.7,
    top_p=1.0,
    max_tokens=500,
    logger=None,
    **kwargs,
):
    seed_instructions_path = "assets/seed_instructions.json"

    with open(seed_instructions_path, "r") as f:
        instruction = json.load(f)

    # logger.info("Completed Instruction Generation")

    machine_output_data = []

    # for instruction in tqdm.tqdm(seed_instructions):
    data = {
        "instruction": instruction["instruction"],
        "input": "",
        "output": "",
        "url": instruction["url"],
    }

    docs = documents_embeds.similarity_search(
        instruction["instruction"], k=num_docs_to_output
    )

    if "summary_embeds" in kwargs:
        with open("assets/vectorstore_summary.pkl", "rb") as f:
            summary_embeds = pickle.load(f)
        docs.extend(
            summary_embeds.similarity_search(
                instruction["instruction"], k=num_docs_to_output
            )
        )

    data["input"] = "\n\n".join([d.page_content for d in docs])
    prompt = encode_prompt_output(
        input_gen=data["input"],
        inst_gen=data["instruction"],
        url_docs=url_docs,
        use_scraped_docs=use_scraped_docs,
    )

    if model_name_code in ["gpt-3.5-turbo", "gpt-4"]:
        max_retries = 10
        retries = 0
        exponential_base = 2
        delay = 1

        while True:
            try:
                code = openai.ChatCompletion.create(
                    model=model_name_code,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                break
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.info(f"Failed after {max_retries} attempts.")
                    raise e
                else:
                    logger.info(
                        f"Attempt {retries} failed with exception: {e}. Retrying..."
                    )
                    delay *= exponential_base * (1 + random.random())

                    time.sleep(delay)

    data["output"] = post_process_response_code(code, model_name_code)
    machine_output_data.append(data)
    return machine_output_data