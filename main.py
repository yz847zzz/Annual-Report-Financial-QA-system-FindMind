import argparse
import json
import os
from datetime import datetime

from loguru import logger

from config import cfg

GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)


def parse_args():
    parser = argparse.ArgumentParser(description='FindMind pipeline entrypoint')
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Run a lightweight sample-data smoke test instead of full model pipeline.',
    )
    return parser.parse_args()


def run_sample_pipeline():
    """Validate bundled sample data and produce a runnable output artifact."""
    required_files = [
        os.path.join(cfg.DATA_PATH, 'test', 'C-list-question.json'),
        os.path.join(cfg.DATA_PATH, 'test', 'C-list-answer.json'),
        os.path.join(cfg.DATA_PATH, 'test', 'output.json'),
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Sample file not found: {path}')

    sample_output_path = os.path.join(cfg.DATA_PATH, 'test', 'output.json')
    with open(sample_output_path, 'r', encoding='utf-8') as f:
        sample_output = json.load(f)

    out_path = os.path.join(cfg.DATA_PATH, 'sample_run_output.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sample_output, f, ensure_ascii=False, indent=2)

    logger.info('Sample mode completed, wrote {} records to {}', len(sample_output), out_path)


def check_paths():
    import torch

    if not os.path.exists(cfg.DATA_PATH):
        raise Exception('DATA_PATH not exists: {}'.format(cfg.DATA_PATH))

    if not os.path.exists(cfg.XPDF_PATH):
        logger.warning('XPDF_PATH not exists, skip xpdf check: {}', cfg.XPDF_PATH)
    else:
        os.system(f"chmod 755 {cfg.XPDF_PATH}/pdftotext")
        os.system(f'{cfg.XPDF_PATH}/pdftotext -table -enc UTF-8 {cfg.DATA_PATH}/check/test.pdf {cfg.DATA_PATH}/check/test.txt')
        with open(f'{cfg.DATA_PATH}/check/test.txt', 'r', encoding='utf-8') as f:
            print(f.readlines()[:10])
        print('Test xpdf success!')

    print('Torch cuda available ', torch.cuda.is_available())

    if not os.path.exists(cfg.CLASSIFY_CHECKPOINT_PATH):
        raise Exception('CLASSIFY_CHECKPOINT_PATH not exists: {}'.format(cfg.CLASSIFY_CHECKPOINT_PATH))
    if not os.path.exists(cfg.NL2SQL_CHECKPOINT_PATH):
        raise Exception('NL2SQL_CHECKPOINT_PATH not exists: {}'.format(cfg.NL2SQL_CHECKPOINT_PATH))
    if not os.path.exists(cfg.KEYWORDS_CHECKPOINT_PATH):
        raise Exception('KEYWORDS_CHECKPOINT_PATH not exists: {}'.format(cfg.KEYWORDS_CHECKPOINT_PATH))

    print('Check paths success!')


def run_full_pipeline():
    # Delay heavy imports so sample mode can run in minimal environments.
    from file import download_data
    from company_table import count_table_keys, build_table
    from chatglm_ptuning import ChatGLM_Ptuning, PtuningType
    from preprocess import extract_pdf_text, extract_pdf_tables
    from check import init_check_dir, check_text, check_tables
    from generate_answer_with_classify import do_gen_keywords
    from generate_answer_with_classify import do_classification, do_sql_generation, generate_answer, make_answer

    # Check whether required directories and data are complete.
    check_paths()

    # 1. Download data into the data directory and generate pdf_info.json.
    download_data()

    # 2. Parse PDFs and extract relevant data.
    extract_pdf_text()
    extract_pdf_tables()

    # 3. Validate extracted data and detect missing items.
    init_check_dir()
    check_text(copy_error_pdf=True)
    check_tables(copy_error_pdf=True)

    # 4. Build the wide company table from extracted fields.
    count_table_keys()
    build_table()

    # 5. Classify user questions.
    model = ChatGLM_Ptuning(PtuningType.Classify)
    do_classification(model)
    model.unload_model()

    # 6. Generate keywords for questions.
    model = ChatGLM_Ptuning(PtuningType.Keywords)
    do_gen_keywords(model)
    model.unload_model()

    # 7. Generate SQL for statistical questions.
    model = ChatGLM_Ptuning(PtuningType.NL2SQL)
    do_sql_generation(model)
    model.unload_model()

    # 8. Generate final answers.
    model = ChatGLM_Ptuning(PtuningType.Nothing)
    generate_answer(model)

    # 9. Export prediction results.
    make_answer()


if __name__ == '__main__':
    args = parse_args()

    DATE = datetime.now().strftime('%Y%m%d')
    log_path = os.path.join(cfg.DATA_PATH, '{}.main.log'.format(DATE))
    if os.path.exists(log_path):
        os.remove(log_path)
    logger.add(log_path, level='DEBUG')

    if args.sample:
        run_sample_pipeline()
    else:
        run_full_pipeline()
