"""
Fashion Semantic Space - Core Functions
时尚语义空间核心功能模块
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# NLP库 - 使用spaCy
import spacy

# 加载spaCy英文模型
nlp = spacy.load('en_core_web_sm')


# ==================== 配置加载 ====================

def load_config(config_path='./config/word_filters.json'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

# 全局配置
CONFIG = None

def init_config(config_path='./config/word_filters.json'):
    """初始化配置"""
    global CONFIG
    CONFIG = load_config(config_path)
    print(f"✓ Configuration loaded from {config_path}")
    return CONFIG


# ==================== 数据加载 ====================

def load_fashion_data(base_path='./AmazonFashionProduct'):
    """加载所有风格的详情和评论数据"""
    
    style_mapping = {
        'athleisure': 'athleisure',
        'bohemian': 'bohemian_fashion',
        'bohemian_fashion': 'bohemian_fashion',
        'casual': 'casual',
        'college': 'college',
        'cute': 'cute',
        'formal': 'formal',
        'gothic': 'gothic',
        'luxury': 'luxury',
        'minimalist': 'minimalist',
        'punk': 'punk',
        'retro': 'retro',
        'romantic': 'romantic',
        'streetwear': 'streetwear',
        'workwear': 'workwear',
        'Y2K': 'Y2K_Aesthetic',
        'Y2K_Aesthetic': 'Y2K_Aesthetic'
    }
    
    all_details = []
    all_reviews = []
    
    # 加载详情数据
    details_path = Path(base_path) / 'details'
    if details_path.exists():
        for style_folder in details_path.iterdir():
            if style_folder.is_dir():
                style_name = style_folder.name.replace('_details', '')
                style_label = style_mapping.get(style_name, style_name)
                
                for csv_file in style_folder.glob('*.csv'):
                    try:
                        df = pd.read_csv(csv_file, encoding='utf-8')
                        df['style'] = style_label
                        df['source'] = 'details'
                        df['file'] = csv_file.name
                        all_details.append(df)
                    except Exception as e:
                        print(f"Error loading {csv_file}: {e}")
    
    # 加载评论数据
    reviews_path = Path(base_path) / 'reviews'
    if reviews_path.exists():
        for style_folder in reviews_path.iterdir():
            if style_folder.is_dir():
                style_name = style_folder.name
                style_label = style_mapping.get(style_name, style_name)
                
                for csv_file in style_folder.glob('*.csv'):
                    try:
                        df = pd.read_csv(csv_file, encoding='utf-8')
                        df['style'] = style_label
                        df['source'] = 'reviews'
                        df['file'] = csv_file.name
                        all_reviews.append(df)
                    except Exception as e:
                        print(f"Error loading {csv_file}: {e}")
    
    details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    reviews_df = pd.concat(all_reviews, ignore_index=True) if all_reviews else pd.DataFrame()
    
    print(f"\n=== Data Loading Summary ===")
    print(f"Total detail records: {len(details_df)}")
    print(f"Total review records: {len(reviews_df)}")
    
    if not details_df.empty:
        print(f"Styles in details: {sorted(details_df['style'].unique())}")
    if not reviews_df.empty:
        print(f"Styles in reviews: {sorted(reviews_df['style'].unique())}")
    
    return details_df, reviews_df


# ==================== 三层文本过滤 ====================

def preprocess_text_basic(text):
    """基础文本预处理"""
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    words = re.findall(r'[a-z]+', text)
    words = [w for w in words if len(w) > 2]
    
    return words


def filter_stopwords(words):
    """第1层：停用词过滤"""
    if CONFIG is None:
        raise ValueError("Config not initialized. Call init_config() first.")
    
    stopwords = set(CONFIG['stopwords'])
    return [w for w in words if w not in stopwords]


def filter_noise_words(words):
    """第2层：噪声词过滤（品牌、平台词等）"""
    if CONFIG is None:
        raise ValueError("Config not initialized. Call init_config() first.")
    
    noise_words = set(CONFIG['noise_words'])
    return [w for w in words if w not in noise_words]


def is_style_word(word):
    """检查是否为风格词"""
    if CONFIG is None:
        return True
    
    style_words_dict = CONFIG['style_words']
    all_style_words = set()
    for category in style_words_dict.values():
        all_style_words.update(category)
    
    return word in all_style_words


def filter_style_words(words, keep_others=True):
    """第3层：风格词提取
    
    Args:
        words: 词列表
        keep_others: 是否保留非风格词（True=宽松模式，False=严格模式）
    """
    if keep_others:
        # 宽松模式：保留所有词，但标记风格词
        return words
    else:
        # 严格模式：只保留风格词
        return [w for w in words if is_style_word(w)]


def preprocess_text_enhanced(text, strict_mode=False):
    """增强版文本预处理（三层过滤）
    
    Args:
        text: 输入文本
        strict_mode: 是否使用严格模式（只保留风格词）
    """
    # 基础预处理
    words = preprocess_text_basic(text)
    
    # 第1层：停用词过滤
    words = filter_stopwords(words)
    
    # 第2层：噪声词过滤
    words = filter_noise_words(words)
    
    # 第3层：风格词提取
    words = filter_style_words(words, keep_others=not strict_mode)
    
    return words


# ==================== 词性标注 ====================

def annotate_pos(words):
    """使用spaCy进行词性标注"""
    if not words:
        return {}
    
    # 将词列表合并为文本
    text = ' '.join(words)
    doc = nlp(text)
    
    # 构建词性字典
    word_pos = {}
    for token in doc:
        if token.text in words:
            word_pos[token.text] = token.pos_
    
    return word_pos


# ==================== 风格词汇提取 ====================

def extract_style_words(details_df, reviews_df, strict_mode=False):
    """从数据中提取风格相关的词汇"""
    style_words = defaultdict(list)
    style_words_pos = defaultdict(dict)  # 存储词性信息
    
    # 处理详情数据
    if not details_df.empty:
        print("\nProcessing details data...")
        detail_columns = ['详情', 'details', 'description', 'Description']
        detail_col = None
        for col in detail_columns:
            if col in details_df.columns:
                detail_col = col
                break
        
        if detail_col:
            for _, row in details_df.iterrows():
                words = preprocess_text_enhanced(row[detail_col], strict_mode)
                style = row['style']
                style_words[style].extend(words)
                
                # 批量词性标注
                if words and len(words) < 100:  # 避免处理太长的文本
                    pos_tags = annotate_pos(words)
                    for word, pos in pos_tags.items():
                        if word not in style_words_pos[style]:
                            style_words_pos[style][word] = pos
            
            print(f"  Processed {len(details_df)} detail records")
    
    # 处理评论数据
    if not reviews_df.empty:
        print("\nProcessing reviews data...")
        review_columns = ['评论详情', 'review', 'Review', 'review_detail']
        review_col = None
        for col in review_columns:
            if col in reviews_df.columns:
                review_col = col
                break
        
        if review_col:
            for _, row in reviews_df.iterrows():
                words = preprocess_text_enhanced(row[review_col], strict_mode)
                style = row['style']
                style_words[style].extend(words)
                
                if words and len(words) < 100:
                    pos_tags = annotate_pos(words)
                    for word, pos in pos_tags.items():
                        if word not in style_words_pos[style]:
                            style_words_pos[style][word] = pos
            
            print(f"  Processed {len(reviews_df)} review records")
    
    return style_words, style_words_pos


# ==================== 风格专属度计算 ====================

def compute_style_specificity(style_words, method='hybrid'):
    """计算每个词对每个风格的专属度
    
    Args:
        style_words: {style: [words]} 字典
        method: 'entropy', 'ratio', 'hybrid'
    
    Returns:
        {style: {word: specificity_score}}
    """
    # 统计词在各风格的频率
    word_style_freq = defaultdict(lambda: defaultdict(int))
    
    for style, words in style_words.items():
        word_counts = Counter(words)
        for word, count in word_counts.items():
            word_style_freq[word][style] = count
    
    # 计算专属度
    specificity = defaultdict(dict)
    
    for word, style_counts in word_style_freq.items():
        total_count = sum(style_counts.values())
        num_styles = len(style_words)
        
        for style in style_words.keys():
            count_in_style = style_counts.get(style, 0)
            
            if count_in_style == 0:
                specificity[style][word] = 0.0
                continue
            
            # 方法1：基于信息熵
            if method in ['entropy', 'hybrid']:
                # 计算词在各风格的概率分布
                probs = []
                for s in style_words.keys():
                    p = style_counts.get(s, 0) / total_count
                    if p > 0:
                        probs.append(p)
                
                # 计算熵
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                max_entropy = np.log2(num_styles) if num_styles > 1 else 1
                entropy_score = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
            
            # 方法2：基于占比
            if method in ['ratio', 'hybrid']:
                ratio_score = count_in_style / total_count
            
            # 综合
            if method == 'hybrid':
                params = CONFIG.get('specificity_params', {})
                alpha = params.get('entropy_weight', 0.6)
                beta = params.get('ratio_weight', 0.4)
                specificity[style][word] = alpha * entropy_score + beta * ratio_score
            elif method == 'entropy':
                specificity[style][word] = entropy_score
            else:  # ratio
                specificity[style][word] = ratio_score
    
    return specificity


# ==================== 词性权重 ====================

def get_pos_weight(word, style, word_pos_dict):
    """获取词性权重
    
    Args:
        word: 词
        style: 风格
        word_pos_dict: {style: {word: pos}}
    """
    if CONFIG is None:
        return 1.0
    
    # 获取词性
    pos = word_pos_dict.get(style, {}).get(word, 'NOUN')
    
    # 尝试使用动态权重（针对特定风格）
    dynamic_weights = CONFIG['pos_weights'].get('dynamic', {})
    if style in dynamic_weights and pos in dynamic_weights[style]:
        return dynamic_weights[style][pos]
    
    # 使用静态权重
    static_weights = CONFIG['pos_weights'].get('static', {})
    return static_weights.get(pos, static_weights.get('default', 1.0))


# ==================== 增强版TF-IDF特征提取 ====================

def extract_enhanced_features(style_words, style_words_pos, max_features=200, min_df=2):
    """增强版特征提取：TF-IDF × 风格专属度 × 词性权重
    
    Returns:
        style_features: {style: [(word, final_score), ...]}
        feature_details: {style: {word: {'tfidf': x, 'specificity': y, 'pos_weight': z}}}
    """
    # 1. 计算基础TF-IDF
    documents = []
    style_names = []
    
    for style, words in style_words.items():
        if words:
            documents.append(' '.join(words))
            style_names.append(style)
    
    if not documents:
        return {}, {}
    
    print(f"\nExtracting enhanced features from {len(documents)} styles...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # 2. 计算风格专属度
    print("Computing style specificity...")
    specificity = compute_style_specificity(style_words)
    
    # 3. 综合计算最终权重
    print("Computing final weighted features...")
    style_features = {}
    feature_details = defaultdict(dict)
    
    for idx, style in enumerate(style_names):
        tfidf_scores = tfidf_matrix[idx].toarray().flatten()
        weighted_features = []
        
        for i, word in enumerate(feature_names):
            if tfidf_scores[i] > 0:
                # 处理可能的词组
                words_in_feature = word.split()
                
                # TF-IDF得分
                tfidf_score = tfidf_scores[i]
                
                # 风格专属度（取平均）
                spec_scores = [specificity[style].get(w, 0.5) for w in words_in_feature]
                spec_score = np.mean(spec_scores) if spec_scores else 0.5
                
                # 词性权重（取平均）
                pos_weights = [get_pos_weight(w, style, style_words_pos) for w in words_in_feature]
                pos_weight = np.mean(pos_weights) if pos_weights else 1.0
                
                # 最终权重
                final_score = tfidf_score * spec_score * pos_weight
                
                weighted_features.append((word, final_score))
                
                # 保存详细信息
                feature_details[style][word] = {
                    'tfidf': float(tfidf_score),
                    'specificity': float(spec_score),
                    'pos_weight': float(pos_weight),
                    'final': float(final_score)
                }
        
        # 排序
        weighted_features.sort(key=lambda x: x[1], reverse=True)
        style_features[style] = weighted_features[:50]
    
    print(f"✓ Enhanced features extracted for {len(style_features)} styles")
    
    return style_features, feature_details


# ==================== 词嵌入 ====================

def create_cooccurrence_matrix(style_words, window_size=5):
    """创建词共现矩阵"""
    vocab = set()
    for words in style_words.values():
        vocab.update(words)
    vocab = sorted(list(vocab))
    vocab_index = {word: i for i, word in enumerate(vocab)}
    
    cooccurrence = np.zeros((len(vocab), len(vocab)))
    
    for style, words in style_words.items():
        for i, word in enumerate(words):
            if word not in vocab_index:
                continue
            
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            
            for j in range(start, end):
                if i != j and words[j] in vocab_index:
                    cooccurrence[vocab_index[word]][vocab_index[words[j]]] += 1
    
    return cooccurrence, vocab, vocab_index


def create_word_embeddings(style_words, n_components=100):
    """创建词嵌入"""
    print(f"\nCreating word embeddings...")
    
    cooccurrence, vocab, vocab_index = create_cooccurrence_matrix(style_words, window_size=5)
    
    if cooccurrence.sum() == 0:
        print("No cooccurrence found")
        return None, None, None
    
    # 应用PPMI
    row_sum = cooccurrence.sum(axis=1, keepdims=True)
    col_sum = cooccurrence.sum(axis=0, keepdims=True)
    total = cooccurrence.sum()
    
    ppmi = np.zeros_like(cooccurrence)
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if cooccurrence[i, j] > 0:
                pmi = np.log2((cooccurrence[i, j] * total) / (row_sum[i] * col_sum[0, j] + 1e-10))
                ppmi[i, j] = max(0, pmi)
    
    # SVD降维
    n_components = min(n_components, min(ppmi.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    word_vectors = svd.fit_transform(ppmi)
    word_vectors = normalize(word_vectors, norm='l2')
    
    word_embeddings = {word: word_vectors[i] for i, word in enumerate(vocab)}
    
    print(f"✓ Created embeddings for {len(word_embeddings)} words with {n_components} dimensions")
    
    return word_embeddings, vocab, vocab_index


# ==================== 风格向量 ====================

def compute_style_vectors(word_embeddings, style_features):
    """计算每个风格的向量（加权平均）"""
    if not word_embeddings or not style_features:
        print("Missing embeddings or features")
        return {}
    
    print("\nComputing style vectors...")
    
    embedding_dim = len(next(iter(word_embeddings.values())))
    style_vectors = {}
    
    for style, features in style_features.items():
        vectors = []
        weights = []
        
        for word, score in features[:30]:
            words = word.split()
            for w in words:
                if w in word_embeddings:
                    vectors.append(word_embeddings[w])
                    weights.append(score)
        
        if vectors:
            vectors = np.array(vectors)
            weights = np.array(weights)
            weights = weights / weights.sum()
            style_vectors[style] = np.average(vectors, axis=0, weights=weights)
        else:
            style_vectors[style] = np.zeros(embedding_dim)
            print(f"Warning: No vectors found for style '{style}'")
    
    print(f"✓ Generated {len(style_vectors)} style vectors")
    
    return style_vectors


# ==================== 相似度分析 ====================

def compute_similarity_matrix(style_vectors):
    """计算风格相似度矩阵"""
    if not style_vectors:
        return pd.DataFrame()
    
    print("\nComputing similarity matrix...")
    
    styles = sorted(style_vectors.keys())
    vectors = np.array([style_vectors[s] for s in styles])
    
    similarity_matrix = cosine_similarity(vectors)
    
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=styles,
        columns=styles
    )
    
    print(f"✓ Similarity matrix computed for {len(styles)} styles")
    
    return similarity_df


def find_style_relationships(similarity_df, top_n=3):
    """找出每个风格最相似和最不相似的风格"""
    print("\n=== Style Relationships ===")
    
    for style in similarity_df.index:
        similarities = similarity_df.loc[style].copy()
        similarities = similarities[similarities.index != style]
        
        most_similar = similarities.nlargest(top_n)
        least_similar = similarities.nsmallest(top_n)
        
        print(f"\n{style}:")
        print(f"  Most similar to:")
        for s, score in most_similar.items():
            print(f"    - {s}: {score:.3f}")
        print(f"  Least similar to:")
        for s, score in least_similar.items():
            print(f"    - {s}: {score:.3f}")


# ==================== 统计展示 ====================

def show_data_statistics(style_words):
    """展示数据统计信息"""
    print("\n=== Style Words Statistics ===")
    total_words = 0
    
    for style, words in sorted(style_words.items()):
        word_count = len(words)
        unique_words = len(set(words))
        total_words += word_count
        
        print(f"\n{style}:")
        print(f"  Total words: {word_count:,}")
        print(f"  Unique words: {unique_words:,}")
        
        if words:
            word_freq = Counter(words)
            top_words = word_freq.most_common(10)
            print(f"  Top 10 words:")
            for word, freq in top_words:
                print(f"    - {word}: {freq}")
    
    print(f"\n=== Total: {total_words:,} words across {len(style_words)} styles ===")


def show_enhanced_features(style_features, feature_details, top_n=15):
    """展示增强版特征（包含详细权重信息）"""
    print("\n=== Enhanced Features by Style ===")
    
    for style in sorted(style_features.keys()):
        features = style_features[style][:top_n]
        print(f"\n{style}:")
        
        for i, (word, final_score) in enumerate(features, 1):
            details = feature_details[style][word]
            print(f"  {i:2d}. {word:25s} | "
                  f"TF-IDF: {details['tfidf']:.3f} × "
                  f"Spec: {details['specificity']:.3f} × "
                  f"POS: {details['pos_weight']:.2f} = "
                  f"{final_score:.4f}")


# ==================== 保存和加载 ====================

def save_all_results(details_df, reviews_df, style_words, style_words_pos,
                     style_features, feature_details, word_embeddings, 
                     style_vectors, similarity_df, base_path='./results'):
    """保存所有分析结果"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(base_path) / timestamp
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Saving results to: {save_path}")
    print(f"{'='*60}")
    
    # 保存数据统计
    data_stats = {
        'details_count': len(details_df),
        'reviews_count': len(reviews_df),
        'styles': list(set(details_df['style'].unique().tolist() + 
                         reviews_df['style'].unique().tolist())) if not details_df.empty or not reviews_df.empty else [],
        'timestamp': timestamp
    }
    with open(save_path / 'data_statistics.json', 'w') as f:
        json.dump(data_stats, f, indent=2)
    
    # 保存风格词汇
    with open(save_path / 'style_words.pkl', 'wb') as f:
        pickle.dump(dict(style_words), f)
    
    # 保存词性标注
    with open(save_path / 'style_words_pos.pkl', 'wb') as f:
        pickle.dump(dict(style_words_pos), f)
    
    # 保存特征（完整版和可读版）
    with open(save_path / 'style_features.pkl', 'wb') as f:
        pickle.dump(style_features, f)
    
    with open(save_path / 'feature_details.json', 'w') as f:
        json.dump(feature_details, f, indent=2)
    
    # 保存词嵌入
    if word_embeddings:
        with open(save_path / 'word_embeddings.pkl', 'wb') as f:
            pickle.dump(word_embeddings, f)
    
    # 保存风格向量
    if style_vectors:
        with open(save_path / 'style_vectors.pkl', 'wb') as f:
            pickle.dump(style_vectors, f)
    
    # 保存相似度矩阵
    if not similarity_df.empty:
        similarity_df.to_csv(save_path / 'similarity_matrix.csv')
        similarity_df.to_pickle(save_path / 'similarity_matrix.pkl')
    
    # 创建摘要报告
    create_summary_report(save_path, style_words, style_features, 
                         feature_details, similarity_df)
    
    print(f"\n✅ All results saved to: {save_path}")
    return save_path


def create_summary_report(save_path, style_words, style_features, 
                          feature_details, similarity_df):
    """创建分析摘要报告"""
    
    report = []
    report.append("="*60)
    report.append("FASHION SEMANTIC SPACE ANALYSIS REPORT")
    report.append("Enhanced with Style Specificity & POS Weighting")
    report.append("="*60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 数据概览
    report.append("DATA OVERVIEW")
    report.append("-"*40)
    report.append(f"Total styles analyzed: {len(style_words)}")
    
    total_words = sum(len(words) for words in style_words.values())
    unique_words = len(set(word for words in style_words.values() for word in words))
    report.append(f"Total words processed: {total_words:,}")
    report.append(f"Unique words: {unique_words:,}\n")
    
    # 每个风格的关键特征
    report.append("KEY FEATURES BY STYLE")
    report.append("-"*40)
    for style in sorted(style_features.keys()):
        features = style_features[style][:5]
        report.append(f"\n{style}:")
        for word, score in features:
            details = feature_details[style][word]
            report.append(f"  • {word:20s} (final: {score:.3f})")
            report.append(f"    TF-IDF: {details['tfidf']:.3f}, "
                        f"Specificity: {details['specificity']:.3f}, "
                        f"POS: {details['pos_weight']:.2f}")
    
    # 风格关系
    if not similarity_df.empty:
        report.append("\n\nSTYLE RELATIONSHIPS")
        report.append("-"*40)
        
        for style in similarity_df.index:
            row = similarity_df.loc[style]
            row = row[row.index != style]
            most_similar = row.nlargest(2)
            
            similar_list = [f"{s} ({score:.2f})" for s, score in most_similar.items()]
            report.append(f"{style}: → {', '.join(similar_list)}")
    
    report_text = '\n'.join(report)
    with open(save_path / 'analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)


def load_results(save_path):
    """加载保存的分析结果"""
    
    save_path = Path(save_path)
    if not save_path.exists():
        print(f"Path {save_path} does not exist")
        return None
    
    results = {}
    
    files_to_load = {
        'style_words': 'style_words.pkl',
        'style_words_pos': 'style_words_pos.pkl',
        'style_features': 'style_features.pkl',
        'word_embeddings': 'word_embeddings.pkl',
        'style_vectors': 'style_vectors.pkl',
        'similarity_matrix': 'similarity_matrix.pkl'
    }
    
    for key, filename in files_to_load.items():
        filepath = save_path / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                results[key] = pickle.load(f)
            print(f"✓ Loaded {key}")
    
    json_files = {
        'data_stats': 'data_statistics.json',
        'feature_details': 'feature_details.json'
    }
    
    for key, filename in json_files.items():
        filepath = save_path / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[key] = json.load(f)
            print(f"✓ Loaded {key}")
    
    return results