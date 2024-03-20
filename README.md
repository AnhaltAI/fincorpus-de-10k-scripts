## Intro
This git repository contains the scripts used to generate the FinCorpus-DE10k dataset, as well as all the plots from the paper. The processing happened in multiple separate steps so the code is relatively messy and spread out through python files and Jupyter Notebooks.


## Basic structure
### Initial building and stats
- `text-extraction-and-stats.py` processed the .PDF files and created .txt files from each when possible, and saved some metadata accessible only from the .PDF such as the number of pages and evtl. PDF metadata.
- `lang-detection.py` does language detection on the txt files using spacy; it also does things like number of tokens/sentences and similar.
- `build-corpus.py` builds the flat directory structure for the dataset, updates the metadata file to use the new paths

### Cleanup and analysis notebooks
- `merge-metadata.ipynb` merges the PDF metadata and the spacy metadata into a single large metadata file
- `dataset-processing.ipynb` does a lot:
	- parse the raw language detection results (probabilities for each of the 20 languages) into a cleaner analyzable representation
	- removes blacklisted files we didn't want to include for various reasons
	- renames collection names
	- removes documents likely to not have been processed correctly based on various heuristics described in the paper,  incl. UTF8
	- does basic stats, removes colums not included in the final dataset 
- `create-safe-dataset.ipynb` removes the two collections that need special handling
- `stats-and-plots.ipynb` contains the analysis of the final dataset and plots used in the paper
- `humaneval.ipynb` contains the trivial calculations of the human evaluated subset of the dataset.
