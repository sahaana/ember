# Ember
Public API for the Ember context enrichment system that enables no-code context enrichment via similarity-based keyless joins. This repository provides a sample implementation of the core functionality described in: https://arxiv.org/abs/2106.01501.  


## Usage
We have provided a demo config (``configs/demo.json``) produced by ``config_gen.ipynb`` that operates over a sample dataset in ``data/abt-buy``. All configuration parameters are provided in ``config_gen.ipynb``, together with high level explanations. To run the demo, simply execute ``python ember.py -c ./configs/demo.json`` from the project home directory. As output, ember prints recall and MRR statistics statistics, and saves the enrichment results to ``data/{data_dir}/results``.

### Custom Dataset
To use Ember on a custom dataset without modification, the dataset must be formatted as in ``data/abt-buy``. The datasets required are as follows:
* ``left.csv``: left dataset, with a header row and id column
* ``right.csv``: right dataset, with a header row and id column
* ``train.pkl``: positive examples only, provided in the form of a list of positive examples
* ``test.pkl``: positive examples only, provided in the form of a list of positive examples

We have provided the original csv files used to generate the processed train and test supervision files as an example. Generating the processed supervision files requires filtering the original supervision files (``*_orig.csv``) to retain only positive examples, and performing a groupby with list aggregation, serialized to preserve lists. 

The public API currently supports one-to-many left and right joins, but additional configuration can be implemented and enabled upon request. We can also provide our supervision processing scripts upon request. 

## Dependencies and Requirements
Dependencies for the public API are as follows: ``jupyter, numpy, scikit-learn, pandas, pytorch, transformers, faiss, rank_bm25, pandarallel``. Experiments for our paper were run on a machine with 504GB of RAM, and four Titan V GPUs with 12GB of memory. We do not recommend running Ember without a GPU. 
