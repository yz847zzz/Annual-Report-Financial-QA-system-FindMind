import os

GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

import torch
from datetime import datetime
from loguru import logger
from config import cfg
from file import download_data
from company_table import count_table_keys, build_table
from qwen_ptuning import QwenLoRA, LoraType
from preprocess import extract_pdf_text, extract_pdf_tables
from check import init_check_dir, check_text, check_tables
from generate_answer_with_classify import do_gen_keywords
from generate_answer_with_classify import do_classification, do_sql_generation, generate_answer, make_answer


def check_paths():
    import ghostscript
    import camelot

    if not os.path.exists(cfg.DATA_PATH):
        raise Exception('DATA_PATH not exists: {}'.format(cfg.DATA_PATH))
    
    if not os.path.exists(cfg.XPDF_PATH):
        raise Exception('XPDF_PATH not exists: {}'.format(cfg.XPDF_PATH))
    else:
        os.chdir(cfg.XPDF_PATH)
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

    """
    for name in ['basic_info', 'employee_info', 'cbs_info', 'cscf_info', 'cis_info', 'dev_info']:
        table_path = os.path.join(cfg.DATA_PATH, '{}.json'.format(name))
        if not os.path.exists(table_path):
            raise Exception('table {} not exists: {}'.format(name, table_path))

    if not os.path.exists(os.path.join(cfg.DATA_PATH, 'CompanyTable.csv')):
        raise Exception('CompanyTable.csv not exists: {}'.format(os.path.join(cfg.DATA_PATH, 'CompanyTable.csv')))
    """
    
    print('Check paths success!')


if __name__ == '__main__':

    # Configure logging.
    DATE = datetime.now().strftime('%Y%m%d')
    log_path = os.path.join(cfg.DATA_PATH, '{}.main.log'.format(DATE))
    if os.path.exists(log_path):
        os.remove(log_path)
    logger.add(log_path, level='DEBUG')

    """
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
    """

    # 5. Classify user questions.
    model = QwenLoRA(LoraType.Classify)
    do_classification(model)

    # 6. Generate keywords for questions.
    model = QwenLoRA(LoraType.Keywords)
    do_gen_keywords(model)

    # 7. Generate SQL for statistical questions.
    model = QwenLoRA(LoraType.NL2SQL)
    do_sql_generation(model)

    # 8. Generate final answers.
    model = QwenLoRA(LoraType.Nothing)
    generate_answer(model)

    # 9. Export prediction results.
    make_answer()
