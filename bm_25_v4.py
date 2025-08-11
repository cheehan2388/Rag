import os
import json
from tqdm import tqdm
import jieba  # 用于中文文本分词
import pdfplumber  # 用于从PDF文件中提取文字的工具
from rank_bm25 import BM25Okapi  # 使用BM25算法进行文件检索
import pytesseract  # 用于OCR识别图片中的文字
from pdf2image import convert_from_path  # 将PDF页面转换为图像
import tempfile  # 用于创建临时文件

# 手設停用詞集合（按類別分開）
STOPWORDS = {
    'finance': {
        "的", "了", "和", "是", "在", "我", "有", "也", "就",
        "不", "人", "都", "一個", "我們", "你", "他", "她",
        "它", "這", "那", "及", "與", "或", "如果",
        "但是", "因為", "所以", "因此", "關於", "通過",
        "此外", "另外", "以及", "等等",
        "金融", "資金",
        # 新增的停用詞
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
        # 新增的停用詞
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
        # 新增的停用詞
        "如何", "多少", "由", "應該", "是否", "什麼時候",
        "怎麼", "怎樣", "什麼", "幾", "哪",
        "需要", "應", "提供", "使用", "可以", "哪裡", "取消"
    }
}

# 载入参考资料，返回一个字典，key为文件名，value为PDF文件内容的文本
def load_data(source_path, category):
    if not os.path.exists(source_path):
        print(f"Source path {source_path} does not exist.")
        return {}
    masked_file_ls = os.listdir(source_path)  # 获取文件夹中的文件列表
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

# 读取单个PDF文件并返回其文本内容，包括表格和图片中的文字
def read_pdf(pdf_loc, category, page_infos: list = None):
    try:
        pdf = pdfplumber.open(pdf_loc)  # 打开指定的PDF文件
    except Exception as e:
        print(f"Failed to open {pdf_loc}: {e}")
        return ''

    # 如果指定了页面范围，则只提取该范围的页面，否则提取所有页面
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
        if word_count > 10:  # 根據提取的文字量決定是否進行OCR
            print(f"Extracted sufficient text from page {page_num + 1} of {pdf_loc}.")
            words = [word for word in jieba.cut_for_search(combined_text) if word not in STOPWORDS[category] and word.strip()]
            pdf_text += ' '.join(words) + ' '
        else:
            print(f"Page {page_num + 1} of {pdf_loc} has insufficient text ({word_count} words), performing OCR on images.")
            try:
                with tempfile.NamedTemporaryFile(suffix='.png') as temp_image:
                    page_image = page.to_image(resolution=300)
                    page_image.save(temp_image.name, format='PNG')
                    ocr_text = pytesseract.image_to_string(temp_image.name, lang='chi_tra')  # 使用繁体中文识别
                    words = [word for word in jieba.cut_for_search(ocr_text) if word not in STOPWORDS[category] and word.strip()]
                    pdf_text += ' '.join(words) + ' '
            except Exception as e:
                print(f"Error during OCR processing of page {page_num + 1} in {pdf_loc}: {e}")
    pdf.close()

    return pdf_text

# 根据查询语句和指定的来源，检索答案
def BM25_retrieve(qs, source, corpus_dict, category):
    # 筛选出源文档中存在的PID
    filtered_corpus = [corpus_dict[int(pid)] for pid in source if int(pid) in corpus_dict]

    if not filtered_corpus:
        print(f"No valid documents found in source: {source}")
        return None

    # 将每篇文档进行分词（已经在read_pdf中完成，并以空格分隔）
    tokenized_corpus = [doc.split(' ') for doc in filtered_corpus]
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25算法建立检索模型

    # 将查询语句进行分词并移除停用词
    tokenized_query = [word for word in jieba.cut_for_search(qs) if word not in STOPWORDS[category] and word.strip()]

    if not tokenized_query:
        print("The query is empty after removing stopwords.")
        return None

    # 根据查询语句检索，返回最相关的文档
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # n为可调整项

    if ans:
        a = ans[0]
        # 找回与最佳匹配文本相对应的文件名
        res = [key for key, value in corpus_dict.items() if value == a]
        return res[0] if res else None  # 返回文件名或None
    else:
        return None

# 主函数
def main(question_path, source_path, output_path):
    answer_dict = {"answers": []}  # 初始化字典

    # 讀取問題文件
    if not os.path.exists(question_path):
        print(f"Question file {question_path} does not exist.")
        return

    with open(question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)  # 读取问题文件

    # 加载各类别的参考资料
    source_path_insurance = os.path.join(source_path, 'insurance')  # 设置参考资料路径
    corpus_dict_insurance = load_data(source_path_insurance, 'insurance')

    source_path_finance = os.path.join(source_path, 'finance')  # 设置参考资料路径
    corpus_dict_finance = load_data(source_path_finance, 'finance')

    # 读取FAQ数据并进行预处理
    faq_path = os.path.join(source_path, 'faq', 'pid_map_content.json')
    if not os.path.exists(faq_path):
        print(f"FAQ file {faq_path} does not exist.")
        corpus_dict_faq = {}
    else:
        with open(faq_path, 'r', encoding='utf-8') as f_s:
            key_to_source_dict = json.load(f_s)  # 读取参考资料文件
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

        # 对FAQ数据进行预处理
        corpus_dict_faq = {}
        for key, value in key_to_source_dict.items():
            # 分词并移除停用词
            words = [word for word in jieba.cut_for_search(str(value)) if word not in STOPWORDS['faq'] and word.strip()]
            corpus_dict_faq[key] = ' '.join(words)

    # 处理每个问题
    for q_dict in tqdm(qs_ref['questions'], desc='Processing questions'):
        qid = q_dict['qid']
        query = q_dict['query']
        source = q_dict['source']
        category = q_dict['category']

        if category == 'finance':
            # 进行检索
            retrieved = BM25_retrieve(query, source, corpus_dict_finance, 'finance')
            # 将结果加入字典
            answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

        elif category == 'insurance':
            retrieved = BM25_retrieve(query, source, corpus_dict_insurance, 'insurance')
            answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

        elif category == 'faq':
            # 只选择当前问题相关的FAQ文档
            corpus_dict_faq_subset = {key: corpus_dict_faq[key] for key in source if key in corpus_dict_faq}
            retrieved = BM25_retrieve(query, source, corpus_dict_faq_subset, 'faq')
            answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

        else:
            print(f"Unknown category '{category}' for QID {qid}. Skipping.")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})

    # 将答案字典保存为json文件
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 保存文件，确保格式和非ASCII字符

    print(f"Retrieval completed. Results saved to {output_path}")

# 設置路徑
question_path = '/content/drive/MyDrive/yushan_data/競賽資料集/dataset/preliminary/questions_example.json'
source_path = '/content/drive/MyDrive/yushan_data/競賽資料集/reference'
output_path = '/content/drive/MyDrive/fintech/answers_bm25_ver_3.json'

# 運行主程序
main(question_path, source_path, output_path)
