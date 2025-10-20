"""
Fashion Semantic Space - Visualization Functions
时尚语义空间可视化模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 相似度热力图 ====================

def plot_similarity_heatmap(similarity_df, figsize=(14, 12), save_path=None):
    """绘制风格相似度热力图"""
    if similarity_df.empty:
        print("No data to plot")
        return
    
    plt.figure(figsize=figsize)
    
    # 使用mask隐藏对角线
    mask = np.zeros_like(similarity_df, dtype=bool)
    np.fill_diagonal(mask, True)
    
    sns.heatmap(
        similarity_df,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Cosine Similarity'},
        mask=mask,
        vmin=-0.2,
        vmax=0.8
    )
    
    plt.title('Fashion Style Similarity Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Style', fontsize=12)
    plt.ylabel('Style', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to {save_path}")
    
    plt.show()


# ==================== 语义空间可视化 ====================

def visualize_style_space(style_vectors, method='pca', figsize=(14, 10), save_path=None):
    """可视化风格语义空间"""
    if not style_vectors or len(style_vectors) < 3:
        print("Not enough data for visualization")
        return
    
    styles = sorted(style_vectors.keys())
    vectors = np.array([style_vectors[s] for s in styles])
    
    # 降维
    if method == 'tsne':
        perplexity = min(30, len(styles) - 1, max(5, len(styles) // 2))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:  # PCA
        reducer = PCA(n_components=2)
    
    coords = reducer.fit_transform(vectors)
    
    # 计算解释的方差（仅对PCA）
    if method == 'pca':
        explained_var = reducer.explained_variance_ratio_
        print(f"Explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")
    
    # 创建颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, len(styles)))
    
    # 绘图
    plt.figure(figsize=figsize)
    
    for i, style in enumerate(styles):
        plt.scatter(coords[i, 0], coords[i, 1], 
                   c=[colors[i]], s=500, alpha=0.7,
                   edgecolors='black', linewidth=1.5)
        
        # 添加标签
        plt.annotate(style, 
                    xy=(coords[i, 0], coords[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=11,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', 
                             alpha=0.7))
    
    plt.title(f'Fashion Style Semantic Space ({method.upper()})', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved semantic space to {save_path}")
    
    plt.show()


# ==================== 特征权重对比图 ====================

def plot_feature_weights_comparison(feature_details, styles=None, top_n=10, save_path=None):
    """对比不同风格的特征权重分解
    
    Args:
        feature_details: {style: {word: {'tfidf': x, 'specificity': y, 'pos_weight': z, 'final': w}}}
        styles: 要显示的风格列表（None则显示全部）
        top_n: 每个风格显示的top特征数
    """
    if not feature_details:
        print("No feature details to plot")
        return
    
    if styles is None:
        styles = sorted(feature_details.keys())[:4]  # 默认显示前4个
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, style in enumerate(styles[:4]):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        details = feature_details[style]
        
        # 获取top N特征
        sorted_features = sorted(details.items(), 
                                key=lambda x: x[1]['final'], 
                                reverse=True)[:top_n]
        
        words = [w for w, _ in sorted_features]
        tfidf_scores = [details[w]['tfidf'] for w, _ in sorted_features]
        spec_scores = [details[w]['specificity'] for w, _ in sorted_features]
        pos_weights = [details[w]['pos_weight'] for w, _ in sorted_features]
        
        x = np.arange(len(words))
        width = 0.25
        
        ax.bar(x - width, tfidf_scores, width, label='TF-IDF', alpha=0.8)
        ax.bar(x, spec_scores, width, label='Specificity', alpha=0.8)
        ax.bar(x + width, pos_weights, width, label='POS Weight', alpha=0.8)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Score')
        ax.set_title(f'{style}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(words, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature weights comparison to {save_path}")
    
    plt.show()


# ==================== 风格专属度热力图 ====================

def plot_specificity_heatmap(feature_details, top_n=15, save_path=None):
    """绘制词的风格专属度热力图
    
    展示哪些词对哪些风格最专属
    """
    if not feature_details:
        print("No feature details to plot")
        return
    
    # 收集所有风格的top词
    all_words = set()
    for style, details in feature_details.items():
        sorted_features = sorted(details.items(), 
                                key=lambda x: x[1]['specificity'], 
                                reverse=True)[:top_n]
        all_words.update([w for w, _ in sorted_features])
    
    all_words = sorted(all_words)[:30]  # 限制显示数量
    styles = sorted(feature_details.keys())
    
    # 构建矩阵
    matrix = np.zeros((len(all_words), len(styles)))
    
    for i, word in enumerate(all_words):
        for j, style in enumerate(styles):
            if word in feature_details[style]:
                matrix[i, j] = feature_details[style][word]['specificity']
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        matrix,
        xticklabels=styles,
        yticklabels=all_words,
        cmap='YlOrRd',
        annot=False,
        cbar_kws={'label': 'Style Specificity'},
        linewidths=0.5
    )
    
    plt.title('Feature Word Style Specificity', fontsize=16, fontweight='bold')
    plt.xlabel('Style', fontsize=12)
    plt.ylabel('Feature Words', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved specificity heatmap to {save_path}")
    
    plt.show()


# ==================== 词频分布图 ====================

def plot_word_frequency_distribution(style_words, styles=None, top_n=20, save_path=None):
    """绘制各风格的词频分布"""
    from collections import Counter
    
    if styles is None:
        styles = sorted(style_words.keys())[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, style in enumerate(styles[:4]):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        words = style_words[style]
        word_freq = Counter(words)
        top_words = word_freq.most_common(top_n)
        
        words_list = [w for w, _ in top_words]
        freqs = [f for _, f in top_words]
        
        ax.barh(range(len(words_list)), freqs, alpha=0.8)
        ax.set_yticks(range(len(words_list)))
        ax.set_yticklabels(words_list)
        ax.set_xlabel('Frequency')
        ax.set_title(f'{style} - Top {top_n} Words', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved word frequency distribution to {save_path}")
    
    plt.show()


# ==================== 风格关系网络图 ====================

def plot_style_network(similarity_df, threshold=0.3, save_path=None):
    """绘制风格关系网络图
    
    Args:
        similarity_df: 相似度矩阵
        threshold: 相似度阈值，只显示高于此值的连接
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed. Install with: pip install networkx")
        return
    
    if similarity_df.empty:
        print("No data to plot")
        return
    
    # 创建图
    G = nx.Graph()
    
    # 添加节点
    styles = similarity_df.index.tolist()
    G.add_nodes_from(styles)
    
    # 添加边（相似度高于阈值的）
    for i, style1 in enumerate(styles):
        for j, style2 in enumerate(styles):
            if i < j:  # 避免重复
                sim = similarity_df.loc[style1, style2]
                if sim > threshold:
                    G.add_edge(style1, style2, weight=sim)
    
    # 绘图
    plt.figure(figsize=(14, 10))
    
    # 使用spring布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 绘制边
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, 
                          width=[w*3 for w in weights],
                          alpha=0.5,
                          edge_color=weights,
                          edge_cmap=plt.cm.Blues)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos,
                          node_size=1000,
                          node_color='lightblue',
                          edgecolors='black',
                          linewidths=2)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos,
                           font_size=10,
                           font_weight='bold')
    
    plt.title(f'Fashion Style Network (similarity > {threshold})', 
             fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved style network to {save_path}")
    
    plt.show()


# ==================== 综合仪表板 ====================

def create_analysis_dashboard(style_vectors, similarity_df, feature_details, 
                              style_words, styles=None, save_path=None):
    """创建综合分析仪表板"""
    
    if styles is None:
        styles = sorted(style_vectors.keys())[:6]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. PCA语义空间（左上，占2x2）
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    plt.sca(ax1)
    
    styles_all = sorted(style_vectors.keys())
    vectors = np.array([style_vectors[s] for s in styles_all])
    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(vectors)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(styles_all)))
    
    for i, style in enumerate(styles_all):
        ax1.scatter(coords[i, 0], coords[i, 1], 
                   c=[colors[i]], s=300, alpha=0.7,
                   edgecolors='black', linewidth=1)
        ax1.annotate(style, xy=(coords[i, 0], coords[i, 1]),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.2', 
                                         facecolor='white', alpha=0.6))
    
    ax1.set_title('Semantic Space (PCA)', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. 相似度热力图（右上）
    ax2 = fig.add_subplot(gs[0:2, 2])
    plt.sca(ax2)
    
    # 选择部分风格的子矩阵
    sub_sim = similarity_df.loc[styles[:6], styles[:6]]
    mask = np.zeros_like(sub_sim, dtype=bool)
    np.fill_diagonal(mask, True)
    
    sns.heatmap(sub_sim, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5,
                mask=mask, vmin=-0.2, vmax=0.8,
                cbar_kws={'label': 'Similarity'}, ax=ax2)
    
    ax2.set_title('Style Similarity', fontweight='bold', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)
    
    # 3. 特征权重对比（底部，占3列）
    for idx in range(3):
        if idx >= len(styles):
            break
        
        ax = fig.add_subplot(gs[2, idx])
        style = styles[idx]
        details = feature_details[style]
        
        sorted_features = sorted(details.items(), 
                                key=lambda x: x[1]['final'], 
                                reverse=True)[:8]
        
        words = [w[:15] for w, _ in sorted_features]  # 截断长词
        finals = [details[w]['final'] for w, _ in sorted_features]
        
        ax.barh(range(len(words)), finals, alpha=0.8, color=colors[idx])
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=8)
        ax.set_xlabel('Final Score', fontsize=9)
        ax.set_title(f'{style}', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    fig.suptitle('Fashion Semantic Space Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved dashboard to {save_path}")
    
    plt.show()