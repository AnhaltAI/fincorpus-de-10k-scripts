"""
Language detection bit.
Provided a directory with txt files:
    - do language detection on the txts
    - count basic stats on the file, e.g. num sentences, num tokens etc.
    - save to a metadata file, with the relative path of the file to the directory 
        is to be used as kindasorta key, later merged to the other metadata files
"""
import logging

from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm
from pprint import pp

from collections import Counter  # , OrderedDict

import pandas as pd

from random import sample

import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector
from spacy.tokens import Doc, Span

from lingua import (
    Language as LinguaLanguage,
    LanguageDetectorBuilder as LinguaLanguageDetectorBuilder,
)
import multiprocessing

# Our code
#  from file_operations import get_file_list_ext

#  logging.basicConfig(encoding="utf-8", level=logging.DEBUG)
logging.basicConfig(encoding="utf-8", level=logging.INFO)
logger = logging.getLogger(__name__)

# Languages from countries from ISINs of the prospectuses
ISIN_LANGUAGES = [
    LinguaLanguage.ENGLISH,
    LinguaLanguage.GERMAN,
    LinguaLanguage.FRENCH,
    LinguaLanguage.DUTCH,
    LinguaLanguage.ITALIAN,
    LinguaLanguage.PORTUGUESE,
    LinguaLanguage.NYNORSK,
    LinguaLanguage.SPANISH,
]

TXT_DIR = Path("/FULL_CLI")

# Possible languages in dataset
LINGUA_LANGUAGES = [
    LinguaLanguage.ENGLISH,
    LinguaLanguage.GERMAN,
]
LINGUA_LANGUAGES = ISIN_LANGUAGES

# Has to be installed like `python -m spacy download de_core_news_lg`
SPACY_MODEL = "de_core_news_lg"

DO_LANGUAGE_DETECTION = True

# Set to True for experimental lingua-py multi-lang-detection
USE_MULTI_LANG_DETECTION = True

# name for 'unknown' language, to make lingua-py compatible with lang-detect
UNKNOWN_KEY = "UNKNOWN"

logger.info(f"Languages: {LINGUA_LANGUAGES}")
logger.info(f"Using lang. detection: {DO_LANGUAGE_DETECTION}")
logger.info(f"\tUsing multi-lang. detection: {USE_MULTI_LANG_DETECTION}")
logger.info(f"Spacy model: {SPACY_MODEL}")


def _get_detector(languages: List[LinguaLanguage] = LINGUA_LANGUAGES):
    detector = LinguaLanguageDetectorBuilder.from_languages(*languages).build()
    return detector


def custom_detection_function(spacy_object):
    """Function that is passed to the language
    detection bit in spacy's pipeline.

    This overrides the default language detection with lingua'b's one,
    because it allows to limit the possible detected languages (in our case,
    to EN and DE).

    Args:
        spacy_object: any spacy object

    Returns:
        Dict[str, Any]: Dictionary containing the language + score
    """
    assert isinstance(spacy_object, Doc) or isinstance(
        spacy_object, Span
    ), "spacy_object must be a spacy Doc or Span object but it is a {}".format(
        type(spacy_object)
    )

    #  languages = [LinguaLanguage.ENGLISH, LinguaLanguage.GERMAN]
    detector = _get_detector()
    lang = detector.detect_language_of(spacy_object.text)

    # Adding dummy score, getting one from linga's detector is easy if needed though
    # If no language found, lingua retuns None, we replace it with "UNKNOWN" to stay compatible with the other lang det
    return {"language": lang.name if lang else UNKNOWN_KEY, "score": 0.95}


def get_lang_detector(nlp, name):
    #  if USE_LINGUEE:
    return LanguageDetector(
        language_detection_function=custom_detection_function, seed=42
    )
    #  else:
    #  return LanguageDetector(seed=42)


nlp_german = spacy.load(SPACY_MODEL)
nlp_german.max_length = 80000000  # some documents are very long
if DO_LANGUAGE_DETECTION and (not USE_MULTI_LANG_DETECTION):
    # if we're doing multi-lang detection, no need to do it inside spacy
    Language.factory("language_detector", func=get_lang_detector)
    nlp_german.add_pipe("language_detector", last=True)


def detect_languages(
    doc,
    multi: bool = USE_MULTI_LANG_DETECTION,
    languages=LINGUA_LANGUAGES,
    num_most_common: int = 3,
) -> Tuple[Dict, Dict]:
    """Returns large and small dict with languages and number of sentences in that language in spacy `doc`
    Args:
        doc:
        multi (bool):
        languages:
        num_most_common (int): size of small dict, excluding OTHER if it's there
            TODO - ugly

    Returns:
        Tuple[Dict, Dict]:
    """
    c = Counter()

    # If we're not doing multi-language detection,  spacy should have this in the pipeline and we just use them
    if not multi:
        logger.debug("Using spacy language detection")
        for sent in doc.sents:
            # print(f"{sent}: {sent._.language}")
            c[sent._.language["language"]] += 1
    else:
        # otherwise we manually detect languages inside doc text
        logger.debug("Using lingua-py detection")
        detector = _get_detector(
            languages=LINGUA_LANGUAGES,
            #  languages=[
            #  LinguaLanguage.GERMAN,
            #  LinguaLanguage.ENGLISH,
            #  LinguaLanguage.UKRAINIAN,
            #  ]
        )

        results = detector.detect_multiple_languages_of(doc.text)

        lang_detected_text_len = 0

        for result in results:
            # Get length of detected text as number of characters
            lang_detected_text_len += result.end_index - result.start_index

            #  detected_text_len = result.end_index - result.start_index

            # Get length of span by number of tokens inside
            # TODO - test how much slower this is
            span = doc.char_span(
                result.start_index,
                result.end_index,
                label=result.language.name,
                alignment_mode="expand",  # get as much as possible
            )

            if span:
                detected_text_len = len(span)
            else:
                # if no valid tokens inside span, consider it not existing
                detected_text_len = 0

            c[result.language.name] += detected_text_len
            logger.debug(
                f"=={result.language.name}==\n {doc.text[result.start_index:result.end_index]}"
            )

            #  if text_length != lang_detected_text_len:
            #  logger.error(f"Length of text and lang.det not matching,{text_length=} {lang_detected_text_len=}")

    # Create both a small and a large normalized dictionary
    # Remove UNKNOWN from smaller one
    most_common_langs = c.most_common(num_most_common + 1)
    small_dict = dict()
    for l in most_common_langs:
        if l[0] != UNKNOWN_KEY:
            small_dict[l[0]] = l[1]

    def _normalize_dict(dd):
        d = dd.copy()
        # normalize languages to %
        sum_of_langs = sum(d.values())

        for k in d.keys():
            d[k] /= sum_of_langs
        return d

    norm_c = _normalize_dict(c)
    norm_s = _normalize_dict(small_dict)
    return norm_c, norm_s


def process_txt(
    path: Path, run_spacy: bool = True, run_ld: bool = DO_LANGUAGE_DETECTION
) -> Dict[str, Any]:
    """Run the entire pipeline (lang.det?, #tokens etc.) on txt file at `path`

    Args:
        path (Path): path to .txt file
        run_spacy (bool): if True, will run spacy
        run_ld: whether to run/use lang det, either as detected by spacy
            or separate multi-language thing
    """
    stats = dict()
    flat_file_path = path.relative_to(TXT_DIR)
    stats["txt_fn"] = str(flat_file_path)
    logger.info(f"processing {stats['txt_fn']} ({TXT_DIR} -> {path})")

    txt = path.read_text(encoding="utf-8")

    stats["num_chars"] = len(txt)
    if run_spacy:
        doc = nlp_german(txt)
        stats["num_tokens"] = len(doc)
        stats["num_sentences"] = len(list(doc.sents))
    if run_ld:
        langs, langs_s = detect_languages(doc)
        stats["langs_raw"], stats["langs_small"] = langs, langs_s
    return stats

def get_file_list_ext(path: Path, limit=0, ext="pdf") -> List[Path]:
    """Given a directory, return a list of max `limit` `ext` files in that
    directory."""
    pdf_list_all = [
        x.resolve() for x in path.glob(f"**/*.{ext}") if f".{ext}" in x.name.lower()
    ]
    pdf_list = sample(
        pdf_list_all, min(len(pdf_list_all), limit) if limit else len(pdf_list_all)
    )
    logger.info(f"Loaded {len(pdf_list)}/{len(pdf_list_all)} files from {str(path)}")
    return pdf_list

def process_directory(path: Path, output_md_path: Path = None, limit=None, **kwargs):
    """Process all txt files in the directory, write results to metadata file
        with each filename written as relative pathto the `path`.

    ! `path` should be the path with first-level directories being collections!

    Args:
        path (Path): path to collections folder
        output_md_path (Path): where to write the metadata csv
        limit: how many files to process, None == all
        kwargs: will be passed directly to process_text
    """
    files = get_file_list_ext(path=path, ext="txt", limit=limit)
    logger.info(f"Beginning processing directory {path} with {len(files)} files.")
    logger.debug(files)
    metadata_results = list()

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    with tqdm(total=len(files), desc='Processing items') as pbar:
        # Use the pool to map the process_item function to the list of items
        results = pool.imap(process_txt, files)
        # Use the update function as callback to update the progress bar
        for result in results:
            metadata_results.append(result)
            pbar.update()

    # Close the pool of processes
    pool.close()
    pool.join()

    df = pd.DataFrame(metadata_results)
    if output_md_path:
        logger.info(f"Wrote metadata CSV to {output_md_path}")
        df.to_csv(output_md_path, index=False)
    return df


def run():
    """
    print(detect_languages("Hier ist mein Text. Here is my text. Ось це мій текст."))

    print(
        detect_languages(
            nlp_german(
                "Das ist mein Text, mit lange deutsche Wörter. Here is my text,  it's clearly english text. Ось це мій текст.\n\n-\n\n\tSome more text.\n\nStop processing here - \n\n\nEND OF TEXT."
            ),
            #  multi=True,
        )
    )

    """

    # ABSOLUTE PATH to directory with .txt file collection
    #   first level subdirs should contain txt files
    #   e.g. if path is /tmp/FULL_CLI , then should contain /tmp/FULL_CLI/Annual_Reports, /tmp/FULL_CLI/Basisprospekte, ... etc.

    # Where to place the CSV file with the result
    OUTPUT_MD_FILE = "/data/FULL_CLI_LD_multi_new.csv"

    LIMIT = 0  # number of files to process, None for all

    # Process and write output file
    md = process_directory(path=TXT_DIR, limit=LIMIT, output_md_path=OUTPUT_MD_FILE)

    print(md)
    #  breakpoint()


if __name__ == "__main__":
    run()
