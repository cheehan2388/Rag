import os
import json
from tqdm import tqdm
import jieba 
import pdfplumber 
from rank_bm25 import BM25Okapi  
import pytesseract   
from pdf2image import convert_from_path  #  
import tempfile 

STOPWORDS = {
    'finance': {
        "的", "了", "和", "是", "在", "我", "有", "也", "就",
        "不", "人", "都", "一個", "我們", "你", "他", "她",
        "它", "這", "那", "及", "與", "或", "如果",
        "但是", "因為", "所以", "因此", "關於", "通過",
        "此外", "另外", "以及", "等等",
        "金融", "資金",
      
        "如何", "多少", "由", "應該", "是否", "什麼時候",
        "怎麼", "怎樣", "什麼", "幾", "哪"
    },
    'insurance': {
        "的", "了", "和", "是", "在", "我", "有", "也", "就",
        "不", "人", "都", "一個", "我們", "你", "他", "她",
        "它", "這", "那", "及", "與", "或", "如果",
        "但是", "因為", "所以", "因此", "關於", "通過",
        "此外", "另外", "以及", "等等",
        "保險", "保單",
    
        "如何", "多少", "由", "應該", "是否", "什麼時候",
        "怎麼", "怎樣", "什麼", "幾", "哪",
        "受益人", "受益", "書面"
    },
    'faq': {
        "的", "了", "和", "是", "在", "我", "有", "也", "就",
        "不", "人", "都", "一個", "我們", "你", "他", "她",
        "它", "這", "那", "及", "與", "或", "如果",
        "但是", "因為", "所以", "因此", "關於", "通過",
        "此外", "另外", "以及", "等等",
        "常見", "問題",
        "如何", "多少", "由", "應該", "是否", "什麼時候",
        "怎麼", "怎樣", "什麼", "幾", "哪",
        "需要", "應", "提供", "使用", "可以", "哪裡", "取消"
    }
}

def load_data(source_path, category):
    if not os.path.exists(source_path):
        print(f"Source path {source_path} does not exist.")
        return {}
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {}
    for file in tqdm(masked_file_ls, desc=f'Loading data from {source_path}'):
        if file.endswith('.pdf'):
            file_path = os.path.join(source_path, file)
            try:
                pid = int(file.replace('.pdf', ''))
            except ValueError:
                print(f"Filename {file} does not match expected pattern. Skipping.")
                continue
            text = read_pdf(file_path, category)
            corpus_dict[pid] = text
    return corpus_dict

def read_pdf(pdf_loc, category, page_infos: list = None):
    try:
        pdf = pdfplumber.open(pdf_loc)
    except Exception as e:
        print(f"Failed to open {pdf_loc}: {e}")
        return ''

    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for page_num, page in enumerate(pages):
        text = page.extract_text()
        tables = page.extract_tables()
        table_text = ''
        if tables:
            for table in tables:
                for row in table:
                    table_text += ' '.join(row) + ' '
        combined_text = (text or '') + ' ' + table_text
        word_count = len(combined_text.strip().split())
        if word_count > 10: 
            print(f"Extracted sufficient text from page {page_num + 1} of {pdf_loc}.")
            words = [word for word in jieba.cut_for_search(combined_text) if word not in STOPWORDS[category] and word.strip()]
            pdf_text += ' '.join(words) + ' '
        else:
            print(f"Page {page_num + 1} of {pdf_loc} has insufficient text ({word_count} words), performing OCR on images.")
            try:
                with tempfile.NamedTemporaryFile(suffix='.png') as temp_image:
                    page_image = page.to_image(resolution=300)
                    page_image.save(temp_image.name, format='PNG')
                    ocr_text = pytesseract.image_to_string(temp_image.name, lang='chi_tra')   
                    words = [word for word in jieba.cut_for_search(ocr_text) if word not in STOPWORDS[category] and word.strip()]
                    pdf_text += ' '.join(words) + ' '
            except Exception as e:
                print(f"Error during OCR processing of page {page_num + 1} in {pdf_loc}: {e}")
    pdf.close()

    return pdf_text

def BM25_retrieve(qs, source, corpus_dict, category):
    filtered_corpus = [corpus_dict[int(pid)] for pid in source if int(pid) in corpus_dict]

    if not filtered_corpus:
        print(f"No valid documents found in source: {source}")
        return None

    tokenized_corpus = [doc.split(' ') for doc in filtered_corpus]
    bm25 = BM25Okapi(tokenized_corpus) 

   
    tokenized_query = [word for word in jieba.cut_for_search(qs) if word not in STOPWORDS[category] and word.strip()]

    if not tokenized_query:
        print("The query is empty after removing stopwords.")
        return None

    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1) 

    if ans:
        a = ans[0]
        res = [key for key, value in corpus_dict.items() if value == a]
        return res[0] if res else None  
    else:
        return None

def main(question_path, source_path, output_path):
    answer_dict = {"answers": []} 

    if not os.path.exists(question_path):
        print(f"Question file {question_path} does not exist.")
        return

    with open(question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f) 

    source_path_insurance = os.path.join(source_path, 'insurance')  
    corpus_dict_insurance = load_data(source_path_insurance, 'insurance')

    source_path_finance = os.path.join(source_path, 'finance') 
    corpus_dict_finance = load_data(source_path_finance, 'finance')

    faq_path = os.path.join(source_path, 'faq', 'pid_map_content.json')
    if not os.path.exists(faq_path):
        print(f"FAQ file {faq_path} does not exist.")
        corpus_dict_faq = {}
    else:
        with open(faq_path, 'r', encoding='utf-8') as f_s:
            key_to_source_dict = json.load(f_s)  
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

        corpus_dict_faq = {}
        for key, value in key_to_source_dict.items():
            words = [word for word in jieba.cut_for_search(str(value)) if word not in STOPWORDS['faq'] and word.strip()]
            corpus_dict_faq[key] = ' '.join(words)

    for q_dict in tqdm(qs_ref['questions'], desc='Processing questions'):
        qid = q_dict['qid']
        query = q_dict['query']
        source = q_dict['source']
        category = q_dict['category']

        if category == 'finance':
            retrieved = BM25_retrieve(query, source, corpus_dict_finance, 'finance')
            answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

        elif category == 'insurance':
            retrieved = BM25_retrieve(query, source, corpus_dict_insurance, 'insurance')
            answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

        elif category == 'faq':
            corpus_dict_faq_subset = {key: corpus_dict_faq[key] for key in source if key in corpus_dict_faq}
            retrieved = BM25_retrieve(query, source, corpus_dict_faq_subset, 'faq')
            answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

        else:
            print(f"Unknown category '{category}' for QID {qid}. Skipping.")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})

    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4) 

    print(f"Retrieval completed. Results saved to {output_path}")

question_path = '/content/drive/MyDrive/yushan_data/競賽資料集/dataset/preliminary/questions_example.json'
source_path = '/content/drive/MyDrive/yushan_data/競賽資料集/reference'
output_path = '/content/drive/MyDrive/fintech/answers_bm25_ver_3.json'

main(question_path, source_path, output_path)