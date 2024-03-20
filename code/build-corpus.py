from typing import List, Tuple, Dict, Optional

from pathlib import Path
from collections import Counter  # , OrderedDict

from tqdm import tqdm

from datetime import datetime
from random import sample
from shutil import copy

import pandas as pd
import numpy as np
import fitz

import json

import logging

logging.basicConfig(encoding="utf-8", level=logging.INFO)

logger = logging.getLogger(__name__)

"""
Concept:
    In: dir with TXT,  dir with PDF,  metadata

    Create new metadata file

    Create output dir
    For each row in metadata:
        - copy txt to new location in dir
        - copy PDF to pdf/collection/xx.pdf
        - append now to metadata file with updated bit

"""


def process_md_corpus(
    md_file: Path,
    txt_dir: Path,
    pdf_dir: Path,
    corpus_dir: Path,
    target_md_file: Path,
    limit: Optional[int] = None,
    version: Optional[str] = "1.0.0",
    md_md: Optional[dict] = None
):
    """Given the metadata file, path to dir with txt files (flattened)
    and path to dir with pdf files (not flattened, paths as in md['source_fn'])
    create a fully flattened corpus and corresponding metadata file.

    Args:
        md_file (Path): md_file
        txt_dir (Path): txt_dir
        pdf_dir (Path): pdf_dir
        corpus_dir (Path): target directory for the corpus 
        target_md_file (Path): location of the new metadata file
        limit (Optional[int]): N of rows/files to process (or None for 'all')
    """
    corpus_dir = corpus_dir.resolve()
    txt_dir = txt_dir.resolve()
    pdf_dir = pdf_dir.resolve()

    target_md_file = target_md_file.resolve()
    target_md_file.parent.mkdir(parents=True, exist_ok=True)

    log_line = f"FROM:\n\t{md_file=}\n\t{txt_dir=}\n\t{pdf_dir=}\nTO:\n\t{corpus_dir=}"
    logger.info(log_line)

    corpus_pdf_loc = corpus_dir / "pdf"
    corpus_txt_loc = corpus_dir / "txt"

    df = (
        pd.read_csv(
            md_file,
            keep_default_na=False,
        )
        #  .convert_dtypes()
        .iloc[:limit]
    )
    #  breakpoint()

    #  shut it up about incompatible dtypes
    #  df["txt_fn"] = df["txt_fn"].astype(str)
    df["pdf_fn"] = ""

    collections = list(df.collection.unique())
    logger.info(f"Collections: {collections}")
    for c in collections:
        (corpus_pdf_loc / c).mkdir(exist_ok=True, parents=True)
        (corpus_txt_loc / c).mkdir(exist_ok=True, parents=True)

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        pdf_location = pdf_dir / row.source_fn

        # just in case
        try:
            assert pdf_location.exists()
        except AssertionError:
            breakpoint()

        fn = pdf_location.name
        collection = row.collection

        new_pdf_location = corpus_pdf_loc / collection / fn

        log_str = f"\t\t{pdf_location} -> {new_pdf_location}"
        logger.debug(log_str)
        copy(pdf_location, new_pdf_location)

        #  has_txt = row.txt_fn is not np.nan
        #  has_txt = row.txt_fn != "nan"
        #  has_txt = row.txt_fn != str(pd.NA)
        has_txt = row.txt_fn != ''

        if has_txt:
            txt_location = txt_dir / row.txt_fn
            assert txt_location.exists()
            new_txt_location = corpus_txt_loc / collection / (fn + ".txt")
            logger.debug(f"\t\t{txt_location} -> {new_txt_location}")
            copy(txt_location, new_txt_location)

        #  new_row = row.to_dict()
        #  new_row.

        # modify metadata in-place
        df.loc[i, "pdf_fn"] = str(new_pdf_location.relative_to(corpus_dir))
        if has_txt:
            df.loc[i, "txt_fn"] = str(new_txt_location.relative_to(corpus_dir))

    meta_metadata = dict()
    meta_metadata['creation_time'] = str(datetime.now())
    meta_metadata['source_txt'] = str(txt_dir)
    meta_metadata['source_pdf'] = str(pdf_dir)
    meta_metadata['source_md'] = str(md_file)
    meta_metadata['version'] = version
    meta_metadata.update(md_md if md_md else {})
    (corpus_dir / "VERSION.json").write_text(json.dumps(meta_metadata), encoding="utf8")

    # TODO - change locations of columns
    cols = list(df.columns)
    cols.remove("source_fn")
    cols.remove("pdf_fn")
    cols.remove("txt_fn")
    cols.remove("collection")

    cols.insert(0, "collection")
    cols.insert(0, "txt_fn")
    cols.insert(0, "pdf_fn")
    cols.insert(-1, "source_fn")

    # Save
    df[cols].to_csv(target_md_file, index_label=False)

    logger.info(f"OUTPUT:\t{target_md_file}")
    #  breakpoint()
    return df


def run():
    # Directory with PDF files
    PDF_DIR = Path("/tmp/BUILD_DS/extracted/")
    # Directory with txt files
    TXT_DIR = Path("/tmp/dataset/flattened/FULL_CLI/")
    # MD
    INPUT_MD_FILE = Path(
        "/data/metadata.csv"
    )

    # Output dir for the entire corpus
    OUTPUT_DIR = Path("/tmp/corpus")
    OUTPUT_MD_FILE_NAME = "metadata.csv"

    LIMIT = 500  # number of files to process, None for all
    LIMIT = None
    process_md_corpus(
        INPUT_MD_FILE,
        txt_dir=TXT_DIR,
        pdf_dir=PDF_DIR,
        corpus_dir=OUTPUT_DIR,
        target_md_file=OUTPUT_DIR / OUTPUT_MD_FILE_NAME,
        limit=LIMIT,
    )


if __name__ == "__main__":
    run()
