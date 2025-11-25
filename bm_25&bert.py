import os
import json
import numpy as np
import torch
from tqdm import tqdm
import jieba
import pdfplumber
from rank_bm25 import BM25Okapi
import pytesseract
from pdf2image import convert_from_path
import tempfile
from sentence_transformers import SentenceTransformer, CrossEncoder, util


EMBEDDING_MODEL_NAME = 'shibing624/text2vec-base-chinese' 
RERANK_MODEL_NAME = 'BAAI/bge-reranker-base'           
TOP_K_RETRIEVE = 50 
TOP_K_RERANK = 1   

STOPWORDS = {
    'finance': {
        "的", "了", "和", "是", "在", "我", "有", "也", "就",
        "不", "人", "都", "一個", "我們", "你", "他", "她",
        "它", "這", "那", "及", "與", "或", "如果",
        "但是", "因為", "所以", "因此", "關於", "通過",
        "此外", "另外", "以及", "等等", "金融", "資金",
        "如何", "多少", "由", "應該", "是否", "什麼時候",
        "怎麼", "怎樣", "什麼", "幾", "哪"
    },
    'insurance': {
        "的", "了", "和", "是", "在", "我", "有", "也", "就",
        "不", "人", "都", "一個", "我們", "你", "他", "她",
        "它", "這", "那", "及", "與", "或", "如果",
        "但是", "因為", "所以", "因此", "關於", "通過",
        "此外", "另外", "以及", "等等", "保險", "保單",
        "如何", "多少", "由", "應該", "是否", "什麼時候",
        "怎麼", "怎樣", "什麼", "幾", "哪",
        "受益人", "受益", "書面"
    },
    'faq': {
        "的", "了", "和", "是", "在", "我", "有", "也", "就",
        "不", "人", "都", "一個", "我們", "你", "他", "她",
        "它", "這", "那", "及", "與", "或", "如果",
        "但是", "因為", "所以", "因此", "關於", "通過",
        "此外", "另外", "以及", "等等", "常見", "問題",
        "如何", "多少", "由", "應該", "是否", "什麼時候",
        "怎麼", "怎樣", "什麼", "幾", "哪",
        "需要", "應", "提供", "使用", "可以", "哪裡", "取消"
    }
}

def read_pdf(pdf_loc, category):

    try:
        pdf = pdfplumber.open(pdf_loc)
    except Exception as e:
        print(f"Failed to open {pdf_loc}: {e}")
        return ''

    pdf_text = ''
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            pdf_text += text
    pdf.close()

    return pdf_text.replace('\n', '').replace(' ', '')

def load_data(source_path, category):

    if not os.path.exists(source_path):
        return {}
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {}
    for file in tqdm(masked_file_ls, desc=f'Loading {category}'):
        if file.endswith('.pdf'):
            pid = int(file.replace('.pdf', ''))
            file_path = os.path.join(source_path, file)
            text = read_pdf(file_path, category)
            corpus_dict[pid] = text
    return corpus_dict


class HybridRetriever:
    def __init__(self, corpus_dict, category):
        self.corpus_dict = corpus_dict
        self.doc_ids = list(corpus_dict.keys())
        self.docs = list(corpus_dict.values())
        self.category = category
       
        print(f"Initializing BM25 for {category}...")
        self.tokenized_docs = [self._tokenize(doc) for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
    
        print(f"Encoding Corpus with BERT for {category}...")
        self.bi_encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.doc_embeddings = self.bi_encoder.encode(self.docs, convert_to_tensor=True, show_progress_bar=True)

    def _tokenize(self, text):
    
        return [word for word in jieba.cut_for_search(text) if word not in STOPWORDS[self.category] and word.strip()]

    def search(self, query, top_k=50, alpha=0.5):

        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
 
        cos_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        bert_scores = cos_scores.cpu().numpy()

        def normalize(scores):
            if np.max(scores) == np.min(scores):
                return scores
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            
        norm_bm25 = normalize(bm25_scores)
        norm_bert = normalize(bert_scores)
        
        hybrid_scores = alpha * norm_bm25 + (1 - alpha) * norm_bert
        
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    
        results = []
        for idx in top_indices:
            results.append({
                "id": self.doc_ids[idx],
                "text": self.docs[idx],
                "score": hybrid_scores[idx]
            })
        return results

class CrossEncoderReranker:
    def __init__(self):
        print("Loading Cross-Encoder Model...")
        self.cross_encoder = CrossEncoder(RERANK_MODEL_NAME)

    def rerank(self, query, candidate_docs):

        if not candidate_docs:
            return None

        pairs = [[query, doc['text']] for doc in candidate_docs]
        
        scores = self.cross_encoder.predict(pairs)
  
        for i, doc in enumerate(candidate_docs):
            doc['rerank_score'] = scores[i]
            
        sorted_docs = sorted(candidate_docs, key=lambda x: x['rerank_score'], reverse=True)
        return sorted_docs


def main(question_path, source_path, output_path):
  
    with open(question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)

 
    retrievers = {}
    
    source_path_insurance = os.path.join(source_path, 'insurance')
    corpus_insurance = load_data(source_path_insurance, 'insurance')
    if corpus_insurance:
        retrievers['insurance'] = HybridRetriever(corpus_insurance, 'insurance')
        
    source_path_finance = os.path.join(source_path, 'finance')
    corpus_finance = load_data(source_path_finance, 'finance')
    if corpus_finance:
        retrievers['finance'] = HybridRetriever(corpus_finance, 'finance')
     
    reranker = CrossEncoderReranker()
    
    answer_dict = {"answers": []}

    for q_dict in tqdm(qs_ref['questions'], desc='Processing Queries'):
        qid = q_dict['qid']
        query = q_dict['query']
        category = q_dict['category']
        source_pids = q_dict['source']  
     
        if category in retrievers:
            retriever = retrievers[category]
            
       
            candidates = retriever.search(query, top_k=TOP_K_RETRIEVE)
            
            
            filtered_candidates = [c for c in candidates if c['id'] in source_pids]
            
          
            if not filtered_candidates:
                filtered_candidates = candidates  
            
            reranked_results = reranker.rerank(query, filtered_candidates)
          
            best_doc_id = reranked_results[0]['id'] if reranked_results else None
            
            answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})
        else:
            
            answer_dict['answers'].append({"qid": qid, "retrieve": None})

 
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
    print(f"Done! Results saved to {output_path}")

 
if __name__ == "__main__":
  
    Q_PATH = 'final_project/questions_example.json'
    SRC_PATH = 'final_project/source' 
    OUT_PATH = 'final_project/answers_hybrid.json'
    
    main(Q_PATH, SRC_PATH, OUT_PATH)