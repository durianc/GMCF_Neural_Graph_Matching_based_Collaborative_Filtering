import numpy as np
import pickle
import os
from tqdm import tqdm

def load_embeddings(embedding_file='../output/embeddings.npy'):
    """
    加载训练过程中保存的嵌入向量
    """
    embeddings = np.load(embedding_file)  # 加载嵌入向量
    return embeddings

def load_dict(file_path):
    """
    加载字典数据（物品或用户字典）
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def recommend(user_id, embeddings, item_dict, user_dict, top_k=5):
    """
    根据给定的 user_id 推荐 top_k 个物品 ID
    :param user_id: 用户的 ID
    :param embeddings: 所有节点（物品和用户）的嵌入向量
    :param item_dict: 物品字典，包含物品 ID 和属性
    :param user_dict: 用户字典，包含用户 ID 和属性
    :param top_k: 推荐的物品数量
    :return: 推荐的物品 ID 列表
    """
    # 获取物品和用户的数量
    num_items = len(item_dict)  # 物品数量
    num_users = len(user_dict)  # 用户数量

    # 输出物品节点和用户节点数量
    print(f"物品节点数量: {num_items}")
    print(f"用户节点数量: {num_users}")

    # 嵌入向量排列：前 num_items 个为物品，接下来的 num_users 个为用户
    # 因此用户的嵌入向量为 embeddings[num_items + user_id]
    user_embedding = embeddings[num_items + user_id]

    # 物品嵌入向量为 embeddings[0:num_items]
    item_embeddings = embeddings[:num_items]

    # 使用向量化计算用户与所有物品的相似度
    print("计算用户与物品的相似度...")
    similarities = np.dot(item_embeddings, user_embedding)

    # 获取与用户最相似的 top_k 个物品的索引
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    # 将物品索引转换为物品 ID，假设字典中保存的顺序与嵌入向量顺序一致
    recommended_item_ids = [list(item_dict.keys())[i] for i in top_k_indices]

    # 输出推荐物品及其与用户的相似度
    print(f"为用户 {user_id} 推荐的物品及其相似度：")
    for idx, item_id in enumerate(recommended_item_ids):
        similarity_percentage = similarities[top_k_indices[idx]] 
        print(f"物品 ID: {item_id}, 相似度: {similarity_percentage:.2f}%")


    return recommended_item_ids


# 示例使用
if __name__ == "__main__":
    # 加载保存的嵌入向量
    embeddings = load_embeddings()

    # 加载物品字典和用户字典
    item_dict = load_dict('../data/parking/item_dict.pkl')
    user_dict = load_dict('../data/parking/user_dict.pkl')

    # 假设我们要为用户 12 推荐 5 个物品
    user_id = 12
    recommended_items = recommend(user_id, embeddings, item_dict, user_dict, top_k=5)

    print(f"为用户 {user_id} 推荐的物品 IDs: {recommended_items}")