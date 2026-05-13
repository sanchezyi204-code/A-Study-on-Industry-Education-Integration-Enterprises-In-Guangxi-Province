import os
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from snownlp import SnowNLP
import re
import gensim
from gensim import corpora,models,similarities
from gensim.models import TfidfModel


# --- 步骤 1: 设置路径和加载停用词 ---
folder_path = r'C:'
stopwords_file_path = r'C:'

# 加载停用词函数
def load_stopwords(filepath):
    """从指定文件路径加载停用词，并返回一个集合(set)。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 读取文件内容，去除每行首尾空格，并放入集合中
            return set([line.strip() for line in f if line.strip()])
    except FileNotFoundError:
        print(f"!!! 警告: 未找到停用词文件: {filepath} !!!")
        print("请检查文件路径和名称。分析将继续，但不会过滤停用词。")
        return set()
    except Exception as e:
        print(f"读取停用词文件时发生错误: {e}")
        return set()

# 调用函数加载停用词
stopwords = load_stopwords(stopwords_file_path)
print(f"已加载 {len(stopwords)} 个停用词。")


# --- 步骤 2: 批量读取、分词和清洗 ---

corpus = []   
all_words_list = [] 
corpus_tokenized = []   # 【新增】存储分词后的列表 (用于共现分析)
print(f"\n开始读取并处理 {folder_path} 路径下的文件...")

# 假设公司报告最多有50份，这里只统计文件数量作为参考
file_count = 0 
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        file_count += 1
        
        # 新增：尝试多种编码读取文件
        encodings = ['utf-8', 'gbk', 'gb18030']  # 定义可能的编码列表
        content = None  # 初始化内容变量
        
        for encoding in encodings:  # 遍历每种编码尝试读取
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()  # 成功读取则赋值给content
                print(f"成功读取 {filename}，使用编码: {encoding}")  # 打印成功信息
                break  # 读取成功就跳出编码尝试循环
            except UnicodeDecodeError:  # 捕获编码错误
                continue  # 继续尝试下一种编码
            except Exception as e:  # 捕获其他异常
                print(f"读取文件 {filename} 时发生错误: {e}")  # 打印错误信息
                break  # 其他错误直接退出
        
        # 新增：检查是否成功读取到内容
        if content is None:  # 如果所有编码都失败
            print(f"无法读取文件 {filename}，跳过此文件。")  # 提示跳过文件
            continue  # 这个continue在外层循环中，是正确的

        # 使用jieba分词
        words = jieba.cut(content, cut_all=False)
        
        # 过滤停用词、单个字符和数字
        filtered_words = [
            word.strip() for word in words 
            if word.strip() not in stopwords and len(word.strip()) > 1 and not word.strip().isdigit()
        ]
        
        # 1. 准备TF-IDF输入 
        corpus.append(" ".join(filtered_words))
        
        # 2. 准备词频统计输入 
        all_words_list.extend(filtered_words)
        
        # LDA
        corpus_tokenized.append(filtered_words)

print(f"成功处理了 {len(corpus)} 份报告。")


## 结果计算与展示

### 1. 词频统计与词云生成

# 词频统计
word_counts = Counter(all_words_list)
top_n = 30

print(f"\n--- 所有报告词频最高的 {top_n} 个词语 ---")
for word, count in word_counts.most_common(top_n):
    print(f"{word}: {count}")

# 词云生成
# 注意：你需要确保这个字体文件存在于你的电脑中，例如 'simhei.ttf' 或 'msyh.ttc'
font_path = 'simhei.ttf' 

try:
    wordcloud = WordCloud(
        font_path=font_path,
        background_color='white',
        width=800,
        height=400,
        max_words=100
    ).generate_from_frequencies(word_counts)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
except FileNotFoundError:
    print(f"!!! 警告: 字体文件 {font_path} 未找到，词云图未生成或可能乱码。!!!")
    
    
    # 创建TF-IDF计算器
vectorizer = TfidfVectorizer()

# 计算TF-IDF矩阵
tfidf_matrix = vectorizer.fit_transform(corpus)

# 获取词汇表
feature_names = vectorizer.get_feature_names_out()

print("\n--- TF-IDF 结果示例 ---")

# 遍历每份报告，找出其TF-IDF值最高的5个关键词
for i, report_tfidf in enumerate(tfidf_matrix.toarray()):
    # 获取该文档中TF-IDF值最高的词语的索引
    # [::-1] 是为了倒序排列，取最大的5个值
    top_indices = report_tfidf.argsort()[-5:][::-1]
    
    # 根据索引获取词语和TF-IDF值
    top_keywords = [feature_names[j] for j in top_indices]
    top_scores = [report_tfidf[j] for j in top_indices]
    
    # 打印结果
    print(f"公司 {i+1} 报告的TF-IDF前5关键词:")
    for kw, score in zip(top_keywords, top_scores):
        print(f"  - {kw} (TF-IDF: {score:.4f})")
    print("-" * 20)

# 计算每个词在所有文档中的平均 TF-IDF 值
avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)

# 创建词语-平均TFIDF对应表
tfidf_dict = dict(zip(feature_names, avg_tfidf))

# 排序并输出前 N 个
top_n = 30
sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

print(f"\n--- 所有报告中平均 TF-IDF 排名前 {top_n} 的词语 ---")
for word, value in sorted_tfidf:
    print(f"{word}: {value:.4f}")

df_top_tfidf = pd.DataFrame(sorted_tfidf, columns=['词语', '平均_TF-IDF'])    
output_path = r'C:\Users\Friday\OneDrive\Desktop\tfidf_top30.xlsx'
df_top_tfidf.to_excel(output_path, index=False)
print(f"\n✅ 已将结果保存至: {output_path}")   