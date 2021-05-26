# SPAINAI HACKATON 2021: NLP CHALLENGE

In this repository I have uploaded all the main code used for obtaining the 3rd prize in the SpainAI Hackaton 2021-NLP Challenge. I haven't had the time to clean the code properly, so please be patient with it, I know some of it might be confusing. I'll try to clean it and organize it better when I find the time for it, so that it's easier to read and use.

## THE CHALLENGE

The challenge consisted on using text descriptions of ZARA products for generating the concrete name of the items, therefore, it's not a text classification problem but a text generation problem. In fact, it can be seen as a summary generation problem. One of the reasons I got involved in this challenge was the difficulty of it, as it's not the usual NLP task.

## MY SOLUTION

My solution can be summarized in the following diagrams, which I will explain step by step, also putting a reference to the scripts where these parts were developed.

### DATA PREPROCESSING AND DATA AUGMENTATION

The first step I took for solving this problem was to get more data from ZARA. For that, I used [zara_crawler.py](zara_crawler.py), which was later updated to [new_zara_crawler.py](new_zara_crawler.py) due to changes in the webpage. These scripts receive a link from a ZARA webpage (for example, for Zara Spain it would be https://www.zara.com/es/en/) and try to search inside each sublink (that represent a product subcategory) for products, getting descriptions and names for those. The crawler is developed using BeautifulSoup and raw requests.
As some categories are not correctly retrieved with this crawler, such as Zara home product pages, I used [another script](crawl_more_zara.py) for scrapping individual links, each representing a products page.
After getting more data to enrich the dataset, I perform a cleaning process, in which I get rid of expressions that can be detrimental for models, such as the height of the models, warnings, promotions expressions, etc. After that, I remove full duplicates (duplicates in name and description). This final part is developed in [process_zara_data.py](process_zara_data.py) This whole process is summarized in the following diagram.

![Alt text](imgs/zara_data.png?raw=true "Data Preparation Process")

### MODELLING PROCESS

In this part, there were too many different things tried, so I just kept here the most relevant ones, those which really helped me to improve the score in the Hackaton. The following diagram summarizes the main parts of this whole process. A I explain these different modelling phases, I'll also refer to the scripts I used for training.

![Alt text](imgs/NLPMODELS.png?raw=True "Data Modelling Process")

The base models used for this were:

* **BART-LARGE**
The performance of the model itself was not the best, as it sometimes produces strange names. However, it was improved using [Population Based Training](https://deepmind.com/blog/article/population-based-training-neural-networks). The main script used for this is [launch_run.py](launch_run.py), although there is another version of it for training with different metrics and a slightly different implementation: [final_train_summarizer.py](final_train_summarizer.py). In both I use Ray's PBT implementation as an integration to Transformers' Trainer. For further improving this model, I decided to re-train its language model, that is, emulate BART's pre-training setup (partially) to help it better learn ZARA data's language distribution, and therefore be able to perform better on the generation step. Texts were corrupted using the functions in [utils_bart_perturbation.py](utils_bart_perturbation.py) with the script [create_data_bart.py](create_data_bart.py).

* **T5-LARGE**: 
This was the model that achieved the best performance among all the individual trained models. It has the peculiarity, shared with Pegasus, that it works much better with Adafactor as the optimizer than with AdamW. For improving the model, I tried to do something similar as with BART, that is, re-training the language model. For that, I used the text perturbation utilities in [utils_perturbation.py](utils_perturbation.py) and just change the imports in [create_data_bart.py](create_data_bart.py). Then, I used the script [basic_training.py](basic_training.py), as with BART, for training the language model. However, when trying to run the generative model from that checkpoint results in Cuda OOM error, even with batch size 1. This effect of increased memory used when re-training the underlying language model was also seen with BART, but as this model is larger, the increase in memory use is impossible to accomodate in a 16GB GPU.

* **T5-3B**:
This model has the same architecture as the previous one, but it has 3B parameters instead of 700M, therefore it's much bigger and should be able to capture much more complicated structures in the data. However, it has the disadvantage that it cannot be trained even with very large gpus (40GB each) unless we do some trick. For that reason I used the competition to investigate the use of [DeepSpeed](https://deepspeed.readthedocs.io/). Using this library we can train models of billions of parameters in not-so-large GPUs, by distributing the parameters, gradients, and optimizer states among the GPUs, and by doing cpu offload. The only disadvantage is that Adafactor is not implemented in the library, therefore we have to use Adam, and that's the reason why this model performs a little bit worse than its smaller cousin. All the code used for training this model and using DeepSpeed to get submissions are in [folder_aws](folder_aws). As you will see, I edited Transformers library, specifically I edited run_seq2seq.py, Seq2SeqTrainer and trainer.py.

* **PEGASUS-LARGE**
This model was trained by Google specifically for summarization, therefore I thought it could fit our problem setting very well. In fact, it almost reaches T5-large performance. 

* **ENCODER-DECODER MODELS**
In [this paper](https://arxiv.org/abs/1907.12461) they describe how we can use pre-trained pure language models such as BERT or RoBERTa to build encoder-decoder architectures useful for text generation tasks. That is exactly what I tried to do, using BERT-large and RoBERTa-large for that. Although the results are not very good, compared to the other models used, the names generated are notably different from those generated by the others, therefore they're good candidates to be part of an Ensemble formed from various models. The training script for Encoder-Decoder Models is [train_encdec.py](train_encdec.py). I also tried to re-train the language model of RoBERTa, but I faced the CUDA OOM issue again, as with T5-large.


* **SUPER NAIVE ENSEMBLE**
This is not an ensemble per se, but a way to put together the knowledge of all my models. For that, I just put in a folder all the submissions I want to put together, then, for each description, I get all the proposed names by all the models included in the ensemble, and sort them by number of occurrences. This way, when a name has been proposed by many models, we assume it's a good candidate name and put it at the beginning. This simple implementation is done in the notebook [SuperNaiveEnsemble.ipynb](SuperNaiveEnsemble.ipynb).

### OTHER THINGS TRIED

* **ProphetNet**
I tried to use Transformers' implementation of ProphetNet, but it has some bugs and doesn't learn anything.

* **ERNIE-GEN**
The results from ERNIE-gen are very promissing, and it seems to be a model at least as good as T5-large looking at the benchmarks. However, the creators decided that neither Tensorflow nor Pytorch were good enough and developed Paddle-Paddle, a library for training deep learning models. In my experience, this library is f***** hell and the model, although public, is not thought for everyone's usage, as the authors have not made any attemt to make the model usable and integrable with the current most used Deep Learning frameworks. I can't say anything good about these guys or their models, I tried to manage their library and code but there were bugs everywhere which were not very explanative, after much effort I had to leave it there.

* **Training a Ranker**
As the metric used for the Hackaton, Discounted Cumulative Gain, is much higher when the first proposals are the ones that match the correct name, I implemented a ranker. In fact, I implemented it in various ways, using Transformers and SentenceTransformers libraries. The implementation for the former is [train_ranker_transformers.py](train_ranker_transformers.py), and for the latter [train_ranker_transformers.py](train_ranker_transformers.py). Sadly, neither of my attempts to train a RoBERTa ranker model were useful, as it was not clearly able to distinguish between good names and bad names, probably because they are really similar, and the model doesn't have enough information in the description to decide which one is good and which one is not.
