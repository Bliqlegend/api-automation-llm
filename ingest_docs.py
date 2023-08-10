"""Load html from files, clean up, split, ingest into FAISS."""
import pickle
from typing import Any, List, Optional, Tuple
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.faiss import FAISS
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import contextlib
import lxml.html as LH
import lxml.html.clean as clean
import requests
import re
import tqdm
import time
import openai


class APIReferenceLoader(WebBaseLoader):
    def __init__(
        self,
        web_path: str,
        header_template: Optional[dict] = None,
        is_visible_scrape: bool = False,
    ):
        super().__init__(web_path=web_path, header_template=header_template)

        self.driver = self.init_firefox_driver()

        self.is_visible_scrape = is_visible_scrape

    def _scrape_bs4(self, url: str) -> Any:
        html_doc = self.session.get(url)
        soup = BeautifulSoup(html_doc.text, "html.parser")
        return soup

    def load(self) -> List[Document]:
        soup = self._scrape_bs4(self.web_path)
        text = soup.get_text()

        metadata = {"source": self.web_path}
        return [Document(page_content=text, metadata=metadata)]

    def find_common_words(self, s, t):
        s_words = s.split()
        t_words = t.split()
        common_words = [word for word in s_words if word in t_words]

        return common_words

    def insert_missing_words(self, s, t, common_words):
        s_words = s.split()
        t_words = t.split()
        missing_words = []

        for i in range(len(common_words) - 1):
            start, end = common_words[i], common_words[i + 1]
            start_idx = s_words.index(start)
            end_idx = s_words.index(end)
            missing_words.extend(s_words[start_idx + 1 : end_idx])

        for word in missing_words:
            if word not in t_words:
                t_words.insert(t_words.index(common_words[-1]) + 1, word)

        t_new = " ".join(t_words)
        return t_new

    def init_firefox_driver(self):
        options = Options()
        options.headless = True
        options.binary = FirefoxBinary(
            "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
            # "/usr/bin/firefox"
        )

        service = FirefoxService(
            executable_path="D:\\2023\\AI\\api-automation-llm\\geckodriver.exe"
            # executable_path="./geckodriver"
        )

        driver = webdriver.Firefox(service=service, options=options)
        return driver

    def scrape_visible_elements(self, url):
        ignore_tags = "style"
        with contextlib.closing(self.driver) as browser:
            browser.get(url)
            time.sleep(10)
            content = browser.page_source
            cleaner = clean.Cleaner()
            content = cleaner.clean_html(content)
            doc = LH.fromstring(content)
            texts = []
            for elt in doc.iterdescendants():
                if elt.tag in ignore_tags:
                    continue
                text = elt.text or ""
                tail = elt.tail or ""
                words = " ".join((text, tail)).strip()
                if words:
                    texts.append(words)
            return " ".join(texts)

    def clean_text(self, text):
        delete_str = "Visible links"
        index = text.find(delete_str)
        if index != -1:
            text = text[:index]
        text = re.sub(r"\n\s*\n", "\n", text.strip())
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"-{3,}", "--", text)
        text = re.sub(r"═{3,}", "==", text)
        text = re.sub(r"_", "", text)
        text = re.sub(r"`", "", text)
        text = re.sub(r"Link: \[\d+\]prefetch", "", text)
        text = re.sub(r"Link: \[\d+\]preload", "", text)
        text = re.sub(r"Link: \[\d+\]preconnect", "", text)
        text = re.sub(r"Link: \[\d+\]canonical", "", text)
        text = re.sub(r"Link: \[\d+\]alternate", "", text)
        text = re.sub(r"\[\d+\]", "", text)
        return text

    def clean_table_content(self, text):
        pass


def hierarchy_links(
    url_docs: str, recursive_depth: int = 1, current_depth: int = 1
) -> List[str]:
    if current_depth > recursive_depth and recursive_depth != 0:
        return []
    elif recursive_depth == 0:
        return [url_docs]

    reqs = requests.get(url_docs)
    soup = BeautifulSoup(reqs.text, "html.parser")
    docs_link = list()
    for link in soup.find_all("a"):
        ref_link = urljoin(url_docs, link.get("href"))
        if url_docs in ref_link and ref_link is not None and url_docs != ref_link:
            docs_link.append(ref_link)
            if current_depth < recursive_depth:
                docs_link.extend(
                    hierarchy_links(ref_link, recursive_depth, current_depth + 1)
                )

    return docs_link


def ingest_docs(
    url_docs: str,
    recursive_depth: int = 1,
    return_summary: bool = True,
    logger=None,
    api_key=None,
) -> Tuple[List, List]:
    openai.api_key = api_key

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docs_link = set(hierarchy_links(url_docs, recursive_depth))
    documents = list()
    docs_for_summary = list()
    logger.info(f"Crawling {docs_link} ...")
    for link in tqdm.tqdm(docs_link):
        loader = APIReferenceLoader(link, is_visible_scrape=True)
        raw_documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=1200, chunk_overlap=150)
        text_splitter_sum = TokenTextSplitter(chunk_size=3100, chunk_overlap=300)
        if return_summary:
            docs_for_summary.extend(text_splitter_sum.split_documents(raw_documents))
        documents.extend(text_splitter.split_documents(raw_documents))
    logger.info("Number of documents: {}".format(len(documents)))

    logger.info("Saving vectorstore into assets/vectorstore.pkl")
    vectorstore = FAISS.from_documents(documents, embeddings)
    with open("assets/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    return documents, docs_for_summary
