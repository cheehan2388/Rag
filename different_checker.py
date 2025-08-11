import json

def find_mismatched_qids(ground_truths_file, answers_file):
    # 讀取 ground_truths 和 answers 文件
    with open(ground_truths_file, 'r') as f:
        ground_truths_data = json.load(f)
    with open(answers_file, 'r') as f:
        answers_data = json.load(f)
    
    # 建立 ground_truths 字典
    ground_truths_dict = {
        item['qid']: {'retrieve': item['retrieve'], 'category': item['category']}
        for item in ground_truths_data['ground_truths']
    }
    
    # 將 answers 轉換為字典，以 qid 為鍵方便查找
    answers_dict = {item['qid']: item['retrieve'] for item in answers_data['answers']}
    
    # 比較 retrieve 值，找出不匹配的 qid
    mismatched_qids = []
    for qid, info in ground_truths_dict.items():
        if qid in answers_dict and answers_dict[qid] != info['retrieve']:
            mismatched_qids.append((qid, info['retrieve'], answers_dict[qid], info['category']))
    
    # 輸出不匹配的 qid
    if mismatched_qids:
        print("Retrieve 不匹配的 qid:")
        for qid, ground_retrieve, answer_retrieve, category in mismatched_qids:
            print(f"qid: {qid}, ground_truth: {ground_retrieve}, answer: {answer_retrieve}, category: {category}")
    else:
        print("所有 retrieve 值都匹配。")

# 使用範例
find_mismatched_qids("ground_truths_example.json", "answers_bm25_v6.json")
