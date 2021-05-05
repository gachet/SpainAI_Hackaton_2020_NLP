import pickle
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Ubuntu/18.04 Chrome 68.0.3409.2 Safari/537.36"

def save_pickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)
        
def clean_text(text):
    text = text.replace("<br/>", "")
    text = text.replace("<br>", "")
    text = text.replace("\xa0", "")
    text = text.replace("•", "")
    text = text.replace("&#39;", "'")
    text = text.replace("\n", "")
    clean_exprs = [
        "HEIGHT OF MODEL",
        "height of model",
        "model height",
        "MODEL HEIGHT",
        "Contains: ",
        "Heel height",
        "Sole height",
        "Height of sole",
        "Height x Length x Width",
        "WARNING",
    ]
    for expr in clean_exprs:
        if expr in text:
            text = text[: text.find(expr)]
    text = text.replace(" +", " ")
    return text


def soup_one_link(link, headers=None, sleep=True):
    if not headers:
        r = requests.get(link)
    else:
        r = requests.get(link, headers=headers)
    print(r)
    if sleep:
        time.sleep(10)
    return BeautifulSoup(r.content, "html.parser")


def google_search(query):
    query = query.replace(" ", "+")
    url = f"https://google.com/search?q={query}"
    return url


class DataGetter:
    """Gets data from different sources related with the test data."""

    def __init__(self, texts, headers={"user-agent": USER_AGENT}):
        self.texts = texts
        self.headers = headers
        self.ignore_expressions = ["webcache.googleusercontent"]
        self.data = {}

    def __call__(
        self,
    ):
        pass

    def _filter_links(self, links):
        """Removes links that have any of ignore_expressions"""
        return [link for link in links if any([badexpr in link for badexpr in self.ignore_expressions])])]

    def _decide_to_stay(self, text, soup_text):
        """If any of the text in soup_text relates to text, it stays. Otherwise, it leaves."""
        soup_text = soup_text.replace("\n", "")
        return text in soup_text

    def _try_exploit_scrap_links(self, text, links):
        """Tries to fetch data from the given links"""
        for link in tqdm(links):
            soup_link = soup_one_link(link, headers=self.headers)
            if self._decide_to_stay(text, soup_link.text):
                pass
            # TODO: CONTINUE FROM HERE.

    def _try_extract_one_item(self, text):
        """
        Receives a text to look for in Google.
        """
        google_query = google_search(text)
        soup_google_query = soup_one_link(google_query, headers=self.headers)
        divisions = soup_google_query.findAll("div", {"class": "tF2Cxc"})
        for division in tqdm(divisions):
            base_link = division.find("div", {"class": "yuRUbf"})
            scrap_links = [
                link["href"]
                for link in base_link.find_all("a", href=True)
            ]
            scrap_links = self._filter_links(scrap_links)
            self._try_exploit_scrap_links(text, scrap_links)
