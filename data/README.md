### Manually created files
- `known_bad_docs.csv` for docs that will get deleted from our dataset in </code/dataset-processing.ipynb>
- `humaneval[_raw].csv` speadsheet with the human evaluation results of a tiny subset of the documents

### Input files needing to be put here
- the results of running </code/text-extraction-and-stats.py> with the PDF extraction and stats 
	- Currenty <./FULL_CLI_2023-09-27T18:08.csv>
- the results of running </code/lang-detection.py> on the txt files, with language detection and spacy stats
	- Currently <./FULL_CLI_LD_multi_new.csv>

They'll be merged into one for the final graphs etc.

### Automatically generated files
- Results of merging PDF and Spacy/LD CSVs:
	- `merge_res_complete.csv` with the merge results for all documents,  including invalid / empty ones, basically with all PDF files
	- **`merge_res.csv` with the merge results of the documents where text was extracted**
- Results of the dataset-processing notebook:
- `metadata_full.csv` - csv with the results of the dataset cleanup, with all details and rows
- **`metadata.csv` - slim CSV with only the necessary details**
	- has empty `txt_fn` for the documents we decided not to include in the text corpus
	- has no:
		- rows of docs we don't include at all
		- columns used as intermediate steps in the notebook
