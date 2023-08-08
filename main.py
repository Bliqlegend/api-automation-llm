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


def create_seed_instructions(user_prompt, api_docs, cfg):
    data = {"id": "seed_instruction_0", "instruction": user_prompt, "url": api_docs}

    if not os.path.exists(cfg.DATA_PATH):
        os.makedirs(cfg.DATA_PATH)

    with open(os.path.join(cfg.DATA_PATH, "seed_instructions.json"), "w") as f:
        json.dump(data, f, indent=4)


def main():
    st.title("API Docs to Code.")

    cfg = OmegaConf.load(os.path.abspath("config.yaml"))
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
                model_name=cfg.OPENAI_ENGINE,
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
