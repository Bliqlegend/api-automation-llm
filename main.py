import streamlit as st
import os
import logging
from omegaconf import OmegaConf
from ingest_docs import ingest_docs
from launch_data_gen import launch_data_gen
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import json
from dotenv import load_dotenv
import time

load_dotenv()

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class SpinnerLogger:
    def __init__(self, message_holder):
        self.message_holder = message_holder

    def info(self, message):
        self.message_holder.text(message)
        time.sleep(2)


spinLogger = SpinnerLogger(st)


def clean_env_if_any():
    if os.path.exists(".env"):
        try:
            os.remove(".env")
        except Exception as e:
            print(f"Error removing .env file: {e}")


def create_seed_instructions(user_prompt, api_docs, cfg):
    data = {"id": "seed_instruction_0", "instruction": user_prompt, "url": api_docs}

    if not os.path.exists(cfg.DATA_PATH):
        os.makedirs(cfg.DATA_PATH)

    with open(os.path.join(cfg.DATA_PATH, "seed_instructions.json"), "w") as f:
        json.dump(data, f, indent=4)


def write_to_env(api_key):
    with open(".env", "w") as f:
        f.write(f'OPENAI_API_KEY="{api_key}"')


models = [
    {"name": "gpt-3.5-turbo", "token_limit": 4096},
    {"name": "gpt-3.5-turbo-0301", "token_limit": 4096},
    {"name": "gpt-3.5-turbo-0613", "token_limit": 4096},
    {"name": "gpt-3.5-turbo-16k", "token_limit": 16384},
    {"name": "gpt-3.5-turbo-16k-0613", "token_limit": 16384},
    {"name": "gpt-4", "token_limit": 8192},
    {"name": "gpt-4-0314", "token_limit": 8192},
    {"name": "gpt-4-0613", "token_limit": 8192},
]


def main():
    clean_env_if_any()
    st.title("API Docs to Code.")

    cfg = OmegaConf.load(os.path.abspath("config.yaml"))

    if st.text_input(
        "OpenAI API Key",
        value=st.session_state.get("OPENAI_API_KEY", ""),
        key="input_OPENAI_API_KEY",
        type="password",
    ):
        if len(st.session_state["input_OPENAI_API_KEY"]) > 0 and st.session_state[
            "input_OPENAI_API_KEY"
        ] != st.session_state.get("OPENAI_API_KEY", ""):
            st.session_state["OPENAI_API_KEY"] = st.session_state[
                "input_OPENAI_API_KEY"
            ]
        write_to_env(st.session_state["OPENAI_API_KEY"])
    model_names = [e["name"] for e in models]
    selected_model = st.selectbox("Select Model", model_names)

    if selected_model:
        st.write(f"You selected: {selected_model}")

        st.session_state["selected_model"] = selected_model

    api_docs = st.text_input("Enter API docs URL here.")
    user_prompt = st.text_input("Describe code you want to generate.")

    if st.button("Generate"):
        if not api_docs or not user_prompt:
            st.warning("Please enter both API docs URL and prompt for code generation.")
            return

        create_seed_instructions(user_prompt, api_docs, cfg)

        message_holder = st.empty()
        spinLogger = SpinnerLogger(message_holder)

        spinLogger.info("Processing..")

        with st.spinner("Doing Stuff"):
            spinLogger.info(
                "Indexing and embedding docs from {api}...".format(api=api_docs)
            )
            logger.info(
                "Indexing and embedding docs from {api}...".format(api=api_docs)
            )

            documents, documents_for_summary = ingest_docs(
                api_docs, recursive_depth=cfg.DEPTH_CRAWLING, logger=logger
            )
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(documents, embeddings)
            logger.info(
                "Done indexing and embedding docs from {api}...".format(api=api_docs)
            )
            spinLogger.info(
                "Done indexing and embedding docs from {api}...".format(api=api_docs)
            )

            logger.info("Launching Code Generation Block...")
            spinLogger.info("Launching Code Generation Block...")

            output_data = launch_data_gen(
                url_docs=api_docs,
                documents_embeds=vectorstore,
                model_name=selected_model,
                logger=logger,
                documents_for_summary=documents_for_summary,
                spinLogger=spinLogger,
            )

            logger.info("Done with Code Generateion Block...")
            spinLogger.info("Done with Code Generateion Block...")
            logger.info("Showing the Code to you....")
            spinLogger.info("Showing the Code to you....")

            st.write("Instruction:", output_data["input"])
            st.write("Here's the optimized code snippet for you.")
            st.code(output_data["output"], language="python")
            st.write("------")

        st.success("Code generated successfully!")

    st.divider()


if __name__ == "__main__":
    main()
