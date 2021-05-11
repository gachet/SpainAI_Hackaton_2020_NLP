# SPAINAI HACKATON 2021: NLP CHALLENGE

In this repository I have uploaded all the main code used for obtaining the 3rd prize in the SpainAI Hackaton 2021-NLP Challenge. I haven't had the time to clean the code properly, so please be patient with it, I know some of it might be confusing. I'll try to clean it and organize it better when I find the time for it, so that it's easier to read and use.

## THE CHALLENGE

The challenge consisted on using text descriptions of ZARA products for generating the concrete name of the items, therefore, it's not a text classification problem but a text generation problem. In fact, it can be seen as a summary generation problem. One of the reasons I got involved in this challenge was the difficulty of it, as it's not the usual NLP task.

## MY SOLUTION

My solution can be summarized in the following diagrams, which I will explain step by step, also putting a reference to the scripts where these parts were developed.

### DATA PREPROCESSING AND DATA AUGMENTATION

The first step I took for solving this problem was to get more data from ZARA. For that, I used [zara_crawler.py](zara_crawler.py), which was later updated to [new_zara_crawler.py](new_zara_crawler.py) due to changes in the webpage. These scripts receive a link from a ZARA webpage (for example, for Zara Spain it would be https://www.zara.com/es/en/) and try to search inside each sublink (that represent a product subcategory) for products, getting descriptions and names for those. The crawler is developed using BeautifulSoup and raw requests.
As some categories are not correctly retrieved with this crawler, such as Zara home product pages, I used [another script](crawl_more_zara.py) for scrapping individual links, each representing a products page.
After getting more data to enrich the dataset, I perform a cleaning process, in which I get rid of expressions that can be detrimental for models, such as the height of the models, warnings, promotions expressions, etc. After that, I remove full duplicates (duplicates in name and description). This whole process is summarized in the following diagram.

![Alt text](imgs/zara_data.png?raw=true "Data Preparation Process")
