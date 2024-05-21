# HeadlineGenerator

This repository holds the notebooks and python scripts used to fine-tune and use the headline generator models we have fine tuned

## Models used
We utilised pre-trained models from Huggingface and fine-tuned them to allow us to generate headlines similar to the ones from our dataset.
- [DistilBART-xsum](https://huggingface.co/sshleifer/distilbart-xsum-12-3) as the BART representative
- [Pegasus-xsum](https://huggingface.co/google/pegasus-xsum) as the PEGASUS representative

Both the above models were trained on the xsum dataset.

## Dataset used

We used the [News Article Category Dataset](https://www.kaggle.com/datasets/timilsinabimal/newsarticlecategories) which consists of 6877 unique datapoints about news articles published in HuffPost, with the topic, person-generated headline and the full news article.

## Tools and Libraries

We used Python as our main scripting language as it has many tools that makes it convenient to work with transformers. Some of which are:
- [Transformers](https://github.com/huggingface/transformers) for dealing with transformers
- [PyTorch](https://pytorch.org/) as a machine learning library
- [Pandas](https://pandas.pydata.org/) for data preprocessing
- [NumPy](https://numpy.org/) for dealing with matrices and math stuff
- [Gradio](https://www.gradio.app/) for generating the user interface

## Running the application
To run the application, run `bash install_requirements.sh`, then run `python app.py`. A server will be set up and run and will be available to use.