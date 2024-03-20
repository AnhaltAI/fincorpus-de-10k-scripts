from typing import List, Tuple, Dict, Optional

from pathlib import Path
from collections import Counter  # , OrderedDict

from tqdm import tqdm

from datetime import datetime
from random import sample

import pandas as pd
import fitz

import logging

logging.basicConfig(encoding="utf-8", level=logging.INFO)

logger = logging.getLogger(__name__)


def get_file_list(path: Path, limit=0) -> List[Path]:
    """Given a directory, return a list of max `limit` PDF files in that
    directory."""
    pdf_list_all = [
        x.resolve() for x in path.glob("**/*.pdf") if ".pdf" in x.name.lower()
    ]
    pdf_list = sample(
        pdf_list_all, min(len(pdf_list_all), limit) if limit else len(pdf_list_all)
    )
    logger.info(f"Loaded {len(pdf_list)}/{len(pdf_list_all)} files from {str(path)}")
    return pdf_list


def analyze_pdf(doc: fitz.Document) -> Dict:
    """Given a fitz Document, return dict with basic PDF stats on it."""
    fs = dict()
    num_pages = len(doc)

    fs["num_pages"] = num_pages
    fs["metadata"] = doc.metadata
    return fs


def process_text_extraction_pdf(
    pdf_path: Path, output_file: Optional[Path] = None
) -> Tuple[str, Dict]:
    """Process a pdf at the pdf_path.

    Return the text and statistics, if `output_file` - save the PDF text there.
    """
    stats = dict()

    # stats["error"] = False
    # stats["is_empty"] = False

    # Full path to document
    stats["doc_path"] = str(pdf_path)

    logger.debug(pdf_path)

    try:
        doc = fitz.open(pdf_path, filetype="pdf")
    except Exception as e:
        logger.error(f"{pdf_path}: {e}")
        stats["error"] = True
        stats["exception_text"] = str(e)
        return None, stats

    # Get the easy stats
    doc_stats = analyze_pdf(doc)
    stats.update(doc_stats)

    # Get text page by page
    logger.debug(stats)
    text = ""
    for page in doc:
        text += page.get_text(sort=True)  # sort=True for text in human reading order

    # Some documents have no OCRd text
    stats["is_empty"] = True if not text else False

    try:
        if output_file and text:
            logger.debug(f"Wrote {output_file}")
            output_file.write_text(text, encoding="utf-8")
            stats["txt_file"] = str(output_file)
    except UnicodeEncodeError as e:
        stats["error"] = True
        stats["exception_text"] = str(e)

    return text, stats


def process_pdf_dir(
    path: Path, output_path: Path, limit: int = 0, **kwargs
) -> List[Dict]:
    """
    Process a directory containing PDFs hidden however many layers deep:
    - Save PDF text in files inside `output_path`,
        flattening the directory structure
    - Return a list of dicts with stats for each of the files

    We assume the directories immediately inside `path` are names of collections,
    for each file its collection name will be added to statistics.
    """

    stats = list()
    pdf_list = get_file_list(path, limit=limit)
    iterable = pdf_list if len(pdf_list) < 50 else tqdm(pdf_list)
    for p in iterable:
        relative_path = p.relative_to(path)

        # Collection name is the first level folder, e.g. "Annual_Reports"
        if relative_path.parts:
            collection_name = relative_path.parts[0]
        else:
            collection_name = "ERROR"  # "financial"
            logger.warning(
                f"Document found in root of relative path: {p}\t{relative_path}"
            )

        # file_output_path = Path(str(output_path / relative_path) + ".txt")
        file_output_path = output_path / collection_name / (p.name + ".txt")

        file_output_path.parent.mkdir(exist_ok=True, parents=True)
        logger.debug(f"{p} / {relative_path} -> {file_output_path}")

        # TODO hail-mary try/catch here
        _, doc_stats = process_text_extraction_pdf(
            pdf_path=p, output_file=file_output_path, **kwargs
        )

        # Add pretty source and output filenames to doc_stats
        doc_stats["source_fn"] = relative_path
        # txt_fn - txt filename - is the key we'll use to merge later
        doc_stats["txt_fn"] = file_output_path.relative_to(output_path)

        if collection_name:
            doc_stats["collection"] = collection_name

        stats.append(doc_stats)
    logger.info(f"Processed {len(stats)} documents")
    return stats


def run_full_analysis_on_dir(
    path: Path,
    output_path: Path,
    limit: int = 0,
    output_metadata_file: Optional[bool | str | Path] = None,
    **kwargs,
):
    """
    Process a directory containing PDFs hidden however many layers deep:
    - Save PDF text in files inside `output_path`,
        flattening the directory structure
    - Return a pandas DataFrame with stats for each file
    - Save the stats to `output_metadata_file` (see source for interpretation)

    We assume the directories immediately inside `path` are names of collections,
    for each file its collection name will be added to statistics.
    """
    logger.info(
        f"\n\tINPUT:\t{str(path)}\n"
        f"\tOUTPUT:\t{str(output_path)}\n"
        f"\tLIMIT:\t{limit}\n"
        f"\tMETADATA:\t{str(output_metadata_file)}\n"
    )
    stats = process_pdf_dir(path=path, output_path=output_path, limit=limit, **kwargs)

    df = pd.DataFrame(stats)

    # if no output file return the dataframe
    if not output_metadata_file:
        return df

    # if it's a boolean true, generate some name and save there
    if output_metadata_file == True:
        output_metadata_file = Path(f"files_{datetime.datetime.now()}.csv")

    # If it's a string, interpret as basename; add date and extension
    if isinstance(output_metadata_file, str):
        output_metadata_file = (
            f"{output_metadata_file}_{datetime.now().isoformat(timespec='minutes')}.csv"
        )

    # Otherwise assume it's a valid path and save there
    Path(output_metadata_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_metadata_file, index=False)

    logger.error(f"Saved to {str(output_metadata_file)}")

    return df


def _sanity_check(df: pd.DataFrame, num: int = 5, path_key="doc_path"):
    """Helper function that outputs the filenames for some files for copypasting"""
    if num is None:
        num = len(df)
        print(num)
    num_to_sample = min(num, len(df))
    for p in df.sample(num_to_sample)[path_key].values:
        print(p)


def run():
    # sanity_check(df[df.is_empty == True], 4)
    # stats = process_pdf_dir(PDF_DIR, Path("/tmp/output"), 100)

    PDF_DIR = Path("/tmp/dataset/extracted/")
    PDF_DIR = Path("/tmp/dataset/extracted/ocrd")
    OUTPUT_DIR = Path("/tmp/remaining_BB_monthly_docs")
    LIMIT = 140  # number of files to process, 0 for all
    LIMIT = 0  # number of files to process, 0 for all
    OUTPUT_MD_FILE = "remaining_BB_monthly"  # will append datetime if str, use directly if Path

    stats = run_full_analysis_on_dir(
        PDF_DIR,
        OUTPUT_DIR,
        LIMIT,
        output_metadata_file=OUTPUT_MD_FILE,  # Path("full.csv")
    )
    print(len(stats))
    print(stats.head())


if __name__ == "__main__":
    run()
