import json

def compare_answers(file1, file2, output_file="mismatched_qids.txt"):
    # 讀取兩個答案文件
    with open(file1, 'r', encoding='utf-8') as f:
        answers_data1 = json.load(f)
    with open(file2, 'r', encoding='utf-8') as f:
        answers_data2 = json.load(f)
    
    # 將每個文件的 answers 轉換為字典，以 qid 為鍵，方便查找
    answers_dict1 = {item['qid']: item['retrieve'] for item in answers_data1['answers']}
    answers_dict2 = {item['qid']: item['retrieve'] for item in answers_data2['answers']}
    
    # 比較 retrieve 值，找出不匹配的 qid
    mismatched_qids = []
    for qid, retrieve1 in answers_dict1.items():
        if qid in answers_dict2:
            retrieve2 = answers_dict2[qid]
            if retrieve1 != retrieve2:
                mismatched_qids.append((qid, retrieve1, retrieve2))
    
    # 將結果寫入txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        if mismatched_qids:
            f.write("Retrieve 不匹配的 qid:\n")
            for qid, retrieve1, retrieve2 in mismatched_qids:
                f.write(f"qid: {qid}, file1 retrieve: {retrieve1}, file2 retrieve: {retrieve2}\n")
        else:
            f.write("所有 retrieve 值都匹配。\n")

# 使用範例
compare_answers("answers1.json", "answers2.json")
