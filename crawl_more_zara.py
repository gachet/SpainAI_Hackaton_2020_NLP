from tqdm import tqdm

from zara_crawler import MyCrawler

links = [
    "https://www.zara.com/uk/en/woman-shoes-l1251.html",
    "https://www.zara.com/us/en/woman-beauty-perfumes-l1415.html",
    "https://www.zara.com/uk/en/home-fragrances-view-all-l4224.html?v1=1682021",
    "https://www.zara.com/uk/en/home-kitchen-dining-l2089.html?v1=1681990",
    "https://www.zara.com/uk/en/home-living-room-l2088.html?v1=1681961",
    "https://www.zara.com/uk/en/home-event-34-l2663.html?v1=1682045",
    "https://www.zara.com/uk/en/woman-bags-l1024.html?v1=1719123",
    "https://www.zara.com/uk/en/man-shoes-l769.html?v1=1720413",
    "https://www.zara.com/uk/en/man-bags-l563.html?v1=1720458",
    "https://www.zara.com/uk/en/woman-accessories-l1003.html?v1=1719379",
    "https://www.zara.com/uk/en/man-accessories-perfumes-l551.html?v1=1720519",
    "https://www.zara.com/uk/en/srpls-man-l1880.html?v1=1720952",
    "https://www.zara.com/uk/en/srpls-woman-l1876.html?v1=1241721",
    "https://www.zara.com/uk/en/woman-accessories-l1003.html?v1=1710185",
    "https://www.zara.com/uk/en/woman-shirts-l1217.html?v1=1718183",
    "https://www.zara.com/uk/en/woman-dresses-l1066.html?v1=1718163",
    "https://www.zara.com/uk/en/woman-outerwear-vests-l1204.html?v1=1718088",
    "https://www.zara.com/uk/en/woman-blazers-l1055.html?v1=1718115",
    "https://www.zara.com/uk/en/woman-trend-1-l1325.html?v1=1235672",
    "https://www.zara.com/uk/en/woman-knitwear-l1152.html?v1=1718198",
    "https://www.zara.com/uk/en/woman-tshirts-l1362.html?v1=1718817",
    "https://www.zara.com/uk/en/woman-body-l1057.html?v1=1718837",
    "https://www.zara.com/uk/en/woman-sweatshirts-l1320.html?v1=1718862",
    "https://www.zara.com/uk/en/woman-jeans-l1119.html?v1=1718780",
    "https://www.zara.com/uk/en/woman-trousers-l1335.html?v1=1718736",
    "https://www.zara.com/uk/en/woman-trousers-shorts-l1355.html?v1=1718879",
    "https://www.zara.com/uk/en/woman-skirts-l1299.html?v1=1718788",
    "https://www.zara.com/uk/en/woman-jackets-l1114.html?v1=1718095",
    "https://www.zara.com/uk/en/woman-outerwear-l1184.html?v1=1718076",
    "https://www.zara.com/uk/en/woman-co-ords-l1061.html?v1=1719476",
    "https://www.zara.com/uk/en/woman-event-2-l1923.html?v1=1719442",
    "https://www.zara.com/uk/en/woman-loungewear-l3519.html?v1=1719457",
    "https://www.zara.com/uk/en/woman-suits-l1437.html?v1=1719408",
    "https://www.zara.com/uk/en/woman-accessories-l1003.html?v1=1719379",
    "https://www.zara.com/uk/en/woman-beauty-perfumes-l1415.html?v1=1719415",
    "https://www.zara.com/uk/en/woman-gift-l4099.html?v1=1719860",
    "https://www.zara.com/uk/en/home-bathroom-l2090.html?v1=1682011",
    "https://www.zara.com/uk/en/home-living-room-l2088.html?v1=1681961",
    "https://www.zara.com/uk/en/home-bedroom-l2087.html?v1=1682005",
    "https://www.zara.com/uk/en/home-homewear-l3018.html?v1=1730778",
    "https://www.zara.com/uk/en/home-kids-new-in-l3974.html?v1=1682031",
    "https://www.zara.com/uk/en/home-fragrances-view-all-l4224.html?v1=1682021",
    "https://www.zara.com/uk/en/home-bathroom-l2090.html?v1=1682011",
    "https://www.zara.com/uk/en/home-kitchen-dining-l2089.html?v1=1681990",
]


if __name__ == "__main__":
    crawler = MyCrawler(
        base_url="www.zara.es",
        datafile="zara_2003.pkl",
        visitedfile="visited_zara_2003.pkl",
    )
    links = list(set(links))
    for link in tqdm(links, desc="Iterating over links"):
        try:
            crawler.crawl_individual_link(link)
        except Exception as e:
            print(e)
            continue
