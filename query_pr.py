import json

def print_all_queries(ground_truths_file):
    # 讀取 ground_truths 文件
    with open(ground_truths_file, 'r',encoding= 'utf-8') as f:
        ground_truths_data = json.load(f)
    
    # 遍歷每個 ground_truth 項目並打印 qid 和 query
    for item in ground_truths_data['questions']:
        qid = item['qid']
        query = item.get('query', "No query provided")  # 如果沒有 query 字段，則顯示 "No query provided"
        print(f"qid: {qid}, query: {query}")

# 使用範例
print_all_queries("questions_example.json")
