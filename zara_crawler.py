import argparse
import os
import pickle
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class MyCrawler:
    def __init__(self, base_url, datafile, visitedfile):
        self.base_url = base_url
        self.datafile = datafile
        self.visitedfile = visitedfile
        # self.saved_products = []  # list of links with products saved.
        self.visited_links = self._load_visited_links(self.visitedfile)
        self.data = self._load_data(self.datafile)
        self.corrupt_name_expressions = [" Details"]

    def __call__(
        self,
    ):
        categories_links = self._categories_links()
        print(categories_links)
        self._try_fetch_from_categories(categories_links)

    def _load_data(self, filename):
        if filename in os.listdir():
            with open(filename, "rb") as f:
                return pickle.load(f)
        else:
            return {"description": [], "name": []}

    def _save_data(
        self,
    ):
        with open(self.datafile, "wb") as f:
            pickle.dump(self.data, f)

    def _load_visited_links(self, filename):
        if filename in os.listdir():
            with open(filename, "rb") as f:
                return pickle.load(f)
        else:
            return []

    def _save_visited_links(
        self,
    ):
        with open(self.visitedfile, "wb") as f:
            pickle.dump(self.visited_links, f)

    def _categories_links(
        self,
    ):
        soup = self.soup_one_link(self.base_url)
        categories = [
            link["href"]
            for link in soup.findAll(
                "a", {"class": "layout-categories-category__link link"}
            )
        ]
        return categories

    def _try_fetch_from_categories(self, categories):
        for category in tqdm(categories, desc="Iterating over categories"):
            try:
                self._try_fetch_one_category(category)
            except Exception as e:
                print(e)
                continue

    def _try_fetch_one_category(self, category_link):
        soup_cat = self.soup_one_link(category_link)
        sublinks = [
            link["href"]
            for link in soup_cat.findAll(
                "a", {"class": "layout-categories-category__link link"}
            )
        ]
        for link in tqdm(sublinks, desc="Iterating over sublinks of category"):
            try:
                soup_link = self.soup_one_link(link)
                self._exploit_one_product_page(soup_link)
            except Exception as e:
                print(e)
                continue

    def _get_products_links_from_link(self, link_souped):
        products_links = [
            link["href"]
            for link in link_souped.findAll(
                "a",
                {"class": "product-link _item product-grid-product-info__name link"},
            )
        ]
        return products_links

    def _exploit_one_product_page(self, link_souped):
        total_links = [link_souped] + self._create_more_links(link_souped)
        for link in tqdm(total_links, desc="Exploitting one product page..."):
            try:
                products_links = self._get_products_links_from_link(link)
                for prod_link in products_links:
                    if prod_link not in self.visited_links:
                        self._get_product_data(prod_link)
                        self._save_data()
                    else:
                        print("Skipping product, as it's already visited and saved")
            except Exception as e:
                print(e)
                continue

    def crawl_individual_link(self, link):
        soup_link = self.soup_one_link(link)
        self._exploit_one_product_page(soup_link)

    def _get_product_data(self, product_link):
        soup_product = self.soup_one_link(product_link)
        product_name = soup_product.find("h1", {"class": "product-name"}).text
        product_name = self._clean_product_name(product_name)
        description = soup_product.find("p", {"class": "description"}).text
        self.data["description"].append(description)
        self.data["name"].append(product_name)
        print(
            f"He conseguido nuevos productos, llevamos {len(self.data['description'])}; {len(self.data['name'])}"
        )

    def _clean_product_name(self, text):
        for corrupt_expr in self.corrupt_name_expressions:
            if corrupt_expr in text:
                text = text.replace(corrupt_expr, "")
        return text

    def _create_more_links(self, link, n=20):
        return [f"{link}&page={i}" for i in range(2, n)]

    def soup_one_link(self, link):
        r = requests.get(
            link,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
            },
        )
        print(r)
        self.visited_links.append(link)
        self._save_visited_links()
        time.sleep(5)
        return BeautifulSoup(r.text, "html.parser")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--link", required=True, type=str, help="Base link to start from"
    )
    parser.add_argument("--datafile", required=True, type=str, help="datafile name")
    parser.add_argument(
        "--visitedfile", required=True, type=str, help="visited links file"
    )
    args = parser.parse_args()
    crawler = MyCrawler(args.link, args.datafile, args.visitedfile)
    crawler()
