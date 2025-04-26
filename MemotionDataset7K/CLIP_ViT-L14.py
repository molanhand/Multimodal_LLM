import os
import copy
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import CLIPProcessor, CLIPModel
import faiss
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# 初始化设置
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 配置参数
class Config:
    # 数据路径配置
    CSV_PATH = "./MemotionDataset7K/memotion_dataset_7k/labels.csv"
    IMAGES_DIR = "./MemotionDataset7K/memotion_dataset_7k/images"
    EMBEDDING_CACHE_DIR = "./MemotionDataset7K/embeddings_cache"
    OUTPUT_DIR = "./MemotionDataset7K/output"
    
    # 模型配置
    MODEL_NAME = "openai/clip-vit-large-patch14"  # 更新为 ViT-L/14
    FREEZE_CLIP = True  # 是否冻结CLIP模型权重
    
    # 训练配置
    BATCH_SIZE = 16 
    NUM_EPOCHS = 15
    LEARNING_RATE = 5e-6 
    WEIGHT_DECAY = 0.01
    TEMPERATURE = 0.07  # 对比损失温度参数
    
    # 检索配置
    RETRIEVAL_K = 5  # 每个样本检索的相似样本数量
    ALPHA = 0.7  # 交叉熵损失和对比损失的权重比例 (动态调整的初始值)
    
    # 数据处理配置
    MAX_TEXT_LENGTH = 77
    USE_DATA_AUGMENTATION = True

# 创建必要目录
os.makedirs(Config.EMBEDDING_CACHE_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# 数据加载和清洗
def load_and_clean_data():
    df = pd.read_csv(Config.CSV_PATH, index_col=0)
    df = df.dropna(subset=["text_corrected", "offensive"])
    df["text_corrected"] = df["text_corrected"].astype(str)
    df = df[df["text_corrected"].str.strip() != ""]
    
    # 验证图像文件
    valid_indices = []
    for i, row in df.iterrows():
        img_path = os.path.join(Config.IMAGES_DIR, row["image_name"])
        try:
            with Image.open(img_path) as im:
                im.verify()
            valid_indices.append(i)
        except Exception as e:
            print(f"损坏文件: {img_path}, 错误: {str(e)}")
    df = df.loc[valid_indices].reset_index(drop=True)
    return df

# 数据预处理
def preprocess_data(df):
    # 合并标签
    def merge_offensive(label):
        if label in ["slight", "very_offensive", "hateful_offensive"]:
            return "offensive"
        return "not_offensive"
    
    df["merge_offensive"] = df["offensive"].apply(merge_offensive)
    label2id = {"not_offensive": 0, "offensive": 1}
    df["label"] = df["merge_offensive"].map(label2id)
    
    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df["label"], random_state=42)
    return train_df, val_df, test_df

# CLIP特征提取函数
def extract_clip_features(dataframe, model_name, device, cache_file=None):
    """提取CLIP特征并缓存结果"""
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        return torch.load(cache_file)
    
    print(f"Extracting CLIP features using {model_name}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_model.eval()
    
    features_list = []
    img_features_list = []
    text_features_list = []
    labels_list = []
    image_names = []
    
    with torch.no_grad():
        for i, row in dataframe.iterrows():
            if i % 100 == 0:
                print(f"Processing {i}/{len(dataframe)}")
            
            img_path = os.path.join(Config.IMAGES_DIR, row["image_name"])
            
            try:
                image = Image.open(img_path).convert("RGB")
                text = row["text_corrected"]
                
                encoded = processor(
                    text=[text],
                    images=[image],
                    padding="max_length",
                    truncation=True,
                    max_length=Config.MAX_TEXT_LENGTH,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(device) for k, v in encoded.items()}
                outputs = clip_model(**inputs)
                
                img_emb = outputs.image_embeds.cpu()
                text_emb = outputs.text_embeds.cpu()
                combined_emb = torch.cat([img_emb, text_emb], dim=1)
                
                features_list.append(combined_emb)
                img_features_list.append(img_emb)
                text_features_list.append(text_emb)
                labels_list.append(row["label"])
                image_names.append(row["image_name"])
            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {str(e)}")
                continue
    
    # 确保有有效的特征
    if not features_list:
        raise ValueError("没有成功提取任何特征，请检查数据路径和图像文件")
            
    features = torch.cat(features_list, dim=0)
    img_features = torch.cat(img_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
    labels = torch.tensor(labels_list)
    
    result = {
        "features": features,
        "img_features": img_features,
        "text_features": text_features,
        "labels": labels,
        "image_names": image_names
    }
    
    if cache_file:
        torch.save(result, cache_file)
        print(f"Saved embeddings to {cache_file}")
    
    return result

# 构建检索索引
def build_faiss_index(features):
    """构建FAISS索引用于高效相似性搜索"""
    print("Building FAISS index...")
    
    # 确保特征是连续的，这对FAISS很重要
    features = features.contiguous()
    
    dim = features.shape[1]
    
    # 使用内积作为相似度度量（余弦相似度）
    index = faiss.IndexFlatIP(dim)
    
    norm_features = F.normalize(features, p=2, dim=1)
    
    # 确保特征是float32类型，这是FAISS要求的
    norm_features_np = norm_features.numpy().astype(np.float32)
    
    index.add(norm_features_np)
    
    return index, norm_features

# 检索相似样本
def retrieve_similar_samples(index, query_features, k=5, exclude_self=True):
    """检索与查询特征最相似的k个样本"""
    norm_query = F.normalize(query_features, p=2, dim=1)
    norm_query_np = norm_query.numpy().astype(np.float32)
    
    # 如果要排除自身，需要多检索一个结果
    actual_k = k + 1 if exclude_self else k
    
    scores, indices = index.search(norm_query_np, actual_k)
    
    # 排除第一个结果（自身）
    if exclude_self:
        indices = indices[:, 1:]
        scores = scores[:, 1:]
    
    return scores, indices

# SimpleContrastiveLoss类实现 
class SimpleContrastiveLoss(nn.Module):
    """简单的对比损失实现"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature # 温度参数

    def forward(self, features, labels, retrieval_features, retrieval_labels):
        """
        计算检索增强的对比损失 (InfoNCE 变体)

        参数:
            features: 当前批次的特征 (锚点), 形状 (batch_size, feature_dim) -> 768 维
            labels: 当前批次的标签, 形状 (batch_size,)
            retrieval_features: 检索到的相似样本特征, 形状 (batch_size, k, feature_dim) -> 768 维
            retrieval_labels: 检索到的样本标签, 形状 (batch_size, k)
        返回:
            torch.Tensor: 计算得到的对比损失标量值
        """
        batch_size = features.size(0)
        feature_dim = features.size(1) # -> 768
        k = retrieval_features.size(1)

        # 可选的维度检查
        if retrieval_features.size(2) != feature_dim:
             # 这个警告现在会检查 768 维
             print(f"警告: SimpleContrastiveLoss 中特征维度不匹配! 锚点: {feature_dim}, 检索到: {retrieval_features.size(2)}")
             return torch.tensor(0.0, device=features.device, requires_grad=True)

        features_norm = F.normalize(features, p=2, dim=1) # (batch_size, feature_dim)

        total_loss = 0.0
        valid_samples_count = 0

        for i in range(batch_size):
            anchor_norm = features_norm[i].unsqueeze(0) # (1, feature_dim)
            anchor_label = labels[i]
            retrieved_batch_norm = F.normalize(retrieval_features[i], p=2, dim=1) # (k, feature_dim)
            retrieved_batch_labels = retrieval_labels[i] # (k,)

            # 计算锚点与所有检索样本的相似度
            similarity_matrix = torch.mm(anchor_norm, retrieved_batch_norm.t()) # (1, feature_dim) @ (feature_dim, k) -> (1, k)
            similarity_matrix = similarity_matrix.squeeze(0) # (k,)
            logits = similarity_matrix / self.temperature

            positive_mask = (retrieved_batch_labels == anchor_label).float()
            valid_retrieval_mask = (retrieved_batch_labels != -1).float()
            positive_mask = positive_mask * valid_retrieval_mask

            if positive_mask.sum() == 0:
                continue

            exp_logits = torch.exp(logits) * valid_retrieval_mask
            log_prob = logits - torch.log(exp_logits.sum(dim=-1, keepdim=True) + 1e-10)
            # InfoNCE 形式：-log( sum(exp(pos))/sum(exp(all)) )
            pos_exp_sum = (exp_logits * positive_mask).sum()
            all_exp_sum = exp_logits.sum()
            loss_anchor = -torch.log(pos_exp_sum / (all_exp_sum + 1e-10) + 1e-10)

            if not torch.isnan(loss_anchor) and not torch.isinf(loss_anchor):
                total_loss += loss_anchor
                valid_samples_count += 1

        avg_loss = total_loss / valid_samples_count if valid_samples_count > 0 else torch.tensor(0.0, device=features.device, requires_grad=True)
        return avg_loss

# ===  HadamardFusion ===
class HadamardFusion(nn.Module): 
    """使用 Hadamard 积 (逐元素乘法) 融合图像和文本嵌入"""
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # self.activation = nn.ReLU() # 可选
        # self.projection = nn.Linear(embed_dim, embed_dim) # 可选
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_emb, text_emb):
        fused = image_emb * text_emb
        fused = self.dropout(fused)
        # fused = self.activation(fused) # 可选
        # fused = self.projection(fused) # 可选
        return fused
# === HadamardFusion 定义结束 ===

# 数据集类增强版
class CLIPMemotionDataset(Dataset):
    def __init__(self, dataframe, images_dir, processor, max_length=77, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        
        # 图像增强
        if self.augment:
            self.image_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image_name"])
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 应用图像增强
            if self.augment:
                image = self.image_transforms(image)
                
            text = row["text_corrected"]
            label = torch.tensor(row["label"], dtype=torch.long)
            
            encoded = self.processor(
                text=[text],
                images=[image],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": encoded["pixel_values"].squeeze(0),
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "label": label,
                "image_name": row["image_name"],
                "text": text
            }
        except Exception as e:
            # 如果处理特定样本出错，返回一个默认项
            print(f"读取样本 {img_path} 时出错: {str(e)}")
            # 重新尝试获取一个有效样本(简单地获取下一个或上一个样本)
            return self.__getitem__((idx + 1) % len(self))

# 改进的模型，更模块化以便于替换CLIP模型
class RGCLCLIPClassifier(nn.Module):
    """基于CLIP的检索增强对比学习分类器"""
    def __init__(self, model_name, num_labels=2, freeze_clip=False):
        super().__init__()
        print(f"初始化 RGCLCLIPClassifier 使用 {model_name}")
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.embed_dim = self.clip_model.config.projection_dim # e.g., 768 for ViT-L/14
        print(f"加载的 CLIP 模型嵌入维度: {self.embed_dim}")

        if freeze_clip:
            print("冻结 CLIP 模型参数。")
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            print("CLIP 模型参数将参与训练。")

        print("使用 Hadamard Product (逐元素乘法) 进行特征融合。")
        self.fusion = HadamardFusion(self.embed_dim)

        # === 修改: 调整分类器中间维度 ===
        # 输入维度现在是 self.embed_dim (768)
        self.classifier = self._create_classifier(self.embed_dim, num_labels)

    def _create_classifier(self, input_dim, num_labels): # input_dim 是 768
        """创建分类器层"""
        # 增大中间层维度以适应更大的输入维度
        hidden_dim1 = 768 # 可以和输入维度一致或稍大/稍小
        hidden_dim2 = 512
        print(f"创建分类器，输入维度: {input_dim}, 中间维度: {hidden_dim1}, {hidden_dim2}, 输出类别数: {num_labels}")
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), # 768 -> 768
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim1, hidden_dim2), # 768 -> 512
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, num_labels) # 512 -> num_labels
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        """模型的前向传播过程."""
        outputs = self.clip_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=False
        )
        image_embeds = outputs.image_embeds # (batch_size, 768)
        text_embeds = outputs.text_embeds   # (batch_size, 768)

        # === 修改: 应用 Hadamard 积融合 ===
        fused_embeds = self.fusion(image_embeds, text_embeds) # 执行逐元素乘法

        logits = self.classifier(fused_embeds)

        return {
            "logits": logits,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "fused_embeds": fused_embeds # -> 768 维
        }

# 提取用于构建索引的融合特征
def extract_fused_features_for_index(model, dataframe, images_dir, processor, device, batch_size=32, cache_file=None):
    """使用模型提取融合特征以构建FAISS索引"""
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached fused embeddings from {cache_file}")
        return torch.load(cache_file)

    print("Extracting fused features for FAISS index...")
    model.eval()

    # 不使用增强
    dataset = CLIPMemotionDataset(dataframe, images_dir, processor, augment=False) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_fused_embeds = []
    all_labels = []
    all_image_names = []

    with torch.no_grad():
        for batch in loader:
            try:
                inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in ["pixel_values", "input_ids", "attention_mask"]
                }
                labels = batch["label"]
                image_names = batch["image_name"]

                outputs = model(**inputs)
                fused_embeds = outputs["fused_embeds"].cpu()

                all_fused_embeds.append(fused_embeds)
                all_labels.append(labels)
                all_image_names.extend(image_names)

            except Exception as e:
                print(f"提取融合特征时出错: {str(e)}")
                continue

    if not all_fused_embeds:
        raise ValueError("未能提取任何融合特征")

    features = torch.cat(all_fused_embeds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    result = {
        "features": features, # 768维的融合特征
        "labels": labels,
        "image_names": all_image_names
    }

    if cache_file:
        torch.save(result, cache_file)
        print(f"Saved fused embeddings to {cache_file}")

    model.train()
    return result

# === apply_smote 使用方差解释率 ===
# SMOTE特征增强核心代码
def apply_smote(features, labels, explained_variance_ratio=0.85): # 添加方差解释率参数
    """
    应用 SMOTE (Synthetic Minority Over-sampling Technique) 进行特征增强.
    在高维特征上先使用 PCA 降维，选择足够解释指定方差比例的最小组件数，
    然后应用 SMOTE，最后通过逆 PCA 转换回原始维度。

    参数:
        features (torch.Tensor 或 np.ndarray): 需要增强的高维特征数据.
        labels (torch.Tensor 或 np.ndarray): 对应的标签数据.
        explained_variance_ratio (float): PCA 降维时希望保留的最小方差解释率 (0 到 1 之间).

    返回:
        tuple: 包含两个 numpy 数组的元组:
               (增强后的特征数据 (原始维度), 增强后的标签数据)
               如果 SMOTE/PCA 无法应用，则返回原始的 features 和 labels (转为 numpy).
    """
    print("开始应用 SMOTE...")
    if isinstance(labels, torch.Tensor):
        labels_np_orig = labels.cpu().numpy()
    else:
        labels_np_orig = labels
    print(f"SMOTE 输入 - 特征形状: {features.shape}, 标签分布: {np.bincount(labels_np_orig)}")

    if isinstance(features, torch.Tensor):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = np.asarray(features)

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    n_samples = features_np.shape[0]
    if n_samples < 2: # PCA和SMOTE至少需要2个样本
        print("警告: 样本数量不足 (< 2)，无法应用 PCA/SMOTE。跳过 SMOTE。")
        return features_np, labels_np

    # 1. PCA 降维 (基于方差解释率)
    # 确保方差阈值在合理范围
    if not 0 < explained_variance_ratio <= 1.0:
        print(f"警告: 无效的 explained_variance_ratio ({explained_variance_ratio})。将使用默认值 0.95。")
        explained_variance_ratio = 0.95

    # === 修改: 使用方差解释率初始化 PCA ===
    print(f"应用 PCA，目标方差解释率: {explained_variance_ratio:.2f}...")
    # 当 n_components 是 float 时，它选择能解释至少这么多方差的组件数
    # 使用 'full' solver 以支持 float n_components
    pca = PCA(n_components=explained_variance_ratio, random_state=42, svd_solver='full') 
    try:
        pca.fit(features_np)
        n_components_selected = pca.n_components_
        print(f"PCA 自动选择 {n_components_selected} 个组件来解释至少 {explained_variance_ratio:.2f} 的方差。")
        # 检查选定的组件数是否足够进行 SMOTE (至少需要 2)
        if n_components_selected < 2:
             print(f"警告: PCA 选择的组件数 ({n_components_selected}) 少于 2，无法应用 SMOTE。跳过 SMOTE。")
             return features_np, labels_np

        low_dim_features = pca.transform(features_np)
        print(f"PCA 降维后特征形状: {low_dim_features.shape}")

    except Exception as e:
        print(f"PCA 拟合或转换过程中出错: {e}. 跳过 SMOTE。")
        return features_np, labels_np # 返回原始数据

    # 2. 应用 SMOTE
    minority_class_label = 1
    minority_count = np.sum(labels_np == minority_class_label)
    k = min(5, minority_count - 1) if minority_count > 1 else 1

    if k < 1:
         print(f"警告: 少数类样本数量 ({minority_count}) 过少，无法设置有效的 k_neighbors。跳过 SMOTE。")
         # 注意：即使跳过 SMOTE，也可能需要返回 PCA 处理后的数据或原始数据
         # 为保持一致性，这里返回原始数据
         return features_np, labels_np

    print(f"应用 SMOTE，少数类样本数: {minority_count}, k_neighbors 设置为: {k}")
    try:
        sm = SMOTE(k_neighbors=k, random_state=42)
        resampled_low_dim, res_labels = sm.fit_resample(low_dim_features, labels_np)
        print(f"SMOTE 后 - 特征形状 (低维): {resampled_low_dim.shape}, 标签分布: {np.bincount(res_labels)}")
    except ValueError as e:
        print(f"SMOTE 应用过程中出错: {e}. 跳过 SMOTE。")
        return features_np, labels_np
    except Exception as e:
        print(f"SMOTE 应用过程中发生未知错误: {e}. 跳过 SMOTE。")
        return features_np, labels_np

    # 3. 添加噪声
    num_original = len(features_np)
    num_synthetic = len(resampled_low_dim) - num_original
    if num_synthetic > 0:
        print(f"为 {num_synthetic} 个合成样本添加高斯噪声...")
        noise_std = 0.01
        noise = np.random.normal(0, noise_std, resampled_low_dim[num_original:].shape)
        resampled_low_dim[num_original:] += noise

    # 4. 逆 PCA 转换
    print("应用逆 PCA 转换回原始维度...")
    try:
        resampled_original_dim = pca.inverse_transform(resampled_low_dim)
    except Exception as e:
        print(f"逆 PCA 转换过程中出错: {e}. 将返回低维度的增强特征。")
        return resampled_low_dim, res_labels # 返回低维数据

    print(f"SMOTE 完成。最终特征形状: {resampled_original_dim.shape}, 最终标签分布: {np.bincount(res_labels)}")
    return resampled_original_dim, res_labels
# === apply_smote 修改结束 ===

# 动态损失加权实现
class DynamicLossWrapper:
    def __init__(self, initial_alpha=0.7, final_alpha=0.9, total_epochs=15):
        """
        Initializes the dynamic loss wrapper. Alpha controls CE loss weight.
        Gradually increases alpha from initial_alpha towards final_alpha.
        """
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_epochs = total_epochs
        # Simple linear interpolation for alpha
        # Could use logarithmic or other schedules too
        self.alpha_schedule = np.linspace(initial_alpha, final_alpha, total_epochs)

    def get_alpha(self, epoch):
        """ Gets the alpha value for the current epoch. """
        # Clip epoch index to be within bounds
        epoch_idx = max(0, min(epoch, self.total_epochs - 1))
        return self.alpha_schedule[epoch_idx]

    def __call__(self, ce_loss, cl_loss, epoch):
        """ Calculates the combined loss using dynamic alpha. """
        current_alpha = self.get_alpha(epoch)
        # Ensure cl_loss is a tensor and on the correct device
        if not isinstance(cl_loss, torch.Tensor):
            cl_loss = torch.tensor(cl_loss, device=ce_loss.device) 
            
        combined_loss = current_alpha * ce_loss + (1.0 - current_alpha) * cl_loss
        # print(f"Epoch {epoch+1}: Alpha={current_alpha:.4f}, CE={ce_loss.item():.4f}, CL={cl_loss.item():.4f}, Combined={combined_loss.item():.4f}") # Debug print
        return combined_loss

# 训练函数 - 使用检索增强的对比学习
def train_rgcl_model(model, train_loader, val_loader, test_loader, device, config=None):
    """训练RGCL模型函数"""
    if config is None:
        config = Config
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.NUM_EPOCHS
    )
    
    # 计算类别权重 (使用原始训练集标签)
    train_labels_list = [item for item in train_df["label"]] 
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.array([0, 1]), 
        y=train_labels_list
    )
    
    criterion_ce = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device)
    )
    criterion_cl = SimpleContrastiveLoss(temperature=config.TEMPERATURE)
    # === Dynamic Loss Wrapper ===
    dynamic_loss = DynamicLossWrapper(
        initial_alpha=config.ALPHA, # 使用config.ALPHA作为起始点
        final_alpha=0.95,           # 逐步增加CE损失的权重
        total_epochs=config.NUM_EPOCHS
    )

    # 在文件名中加入维度和融合方法信息，避免不同模型配置混淆缓存
    model_name_slug = config.MODEL_NAME.replace('/', '_')
    embed_dim = model.embed_dim 
    fusion_method_slug = "hadamard" 
    cache_filename = f"train_fused_features_{model_name_slug}_dim{embed_dim}_{fusion_method_slug}_smote.pt" 
    cache_file = os.path.join(config.EMBEDDING_CACHE_DIR, cache_filename)
    print(f"融合特征缓存文件路径: {cache_file}")

    index = None
    norm_features = None
    retrieval_labels = None

    try:
        # 1. 提取原始训练集的融合特征 (768 维)
        print("步骤 1: 提取原始训练集融合特征...")
        train_fused_features_data = extract_fused_features_for_index(
            model,
            train_df, # 确保 train_df 可用
            Config.IMAGES_DIR,
            train_loader.dataset.processor,
            device,
            batch_size=config.BATCH_SIZE,
            # cache_file=None # 取消注释这行以强制重新提取特征
            cache_file=cache_file 
        )
        original_features = train_fused_features_data["features"] # Tensor (CPU, 768维)
        original_labels = train_fused_features_data["labels"]     # Tensor (CPU)
        print(f"提取到 {len(original_features)} 个 {original_features.shape[1]}维 原始融合特征。") 

        # 2. 对提取到的融合特征应用 SMOTE (使用方差解释率进行 PCA)
        print("步骤 2: 应用 SMOTE (使用方差解释率进行 PCA)...")
        smote_features_np, smote_labels_np = apply_smote(
            original_features,
            original_labels
            # 可以显式传递: explained_variance_ratio=0.9
        )
        print(f"SMOTE 处理后得到 {len(smote_features_np)} 个特征。")

        smote_features = torch.from_numpy(smote_features_np).float()
        smote_labels = torch.from_numpy(smote_labels_np).long()

        # 3. 使用 SMOTE 增强后的特征构建 FAISS 索引 (768 维)
        print("步骤 3: 使用 SMOTE 增强特征构建 FAISS 索引...")
        index, norm_features = build_faiss_index(smote_features) # 处理 768 维特征
        retrieval_labels = smote_labels
        print(f"FAISS 索引构建完成。索引包含 {index.ntotal} 个样本，维度 {index.d}。") # 打印 768
        print(f"索引中标签分布: {np.bincount(retrieval_labels.numpy())}")

        # (可选) 缓存 SMOTE 后的特征/标签
        # torch.save({"features": smote_features, "labels": smote_labels}, cache_file + ".data")


    except Exception as e:
        print(f"在准备检索索引时发生严重错误: {str(e)}")
        print("将无法使用检索增强对比损失，仅使用交叉熵损失进行训练。")
        index = None
        norm_features = None
        retrieval_labels = None

    # === Training Loop Start ===
    best_val_f1 = 0.0
    best_state = None
    patience = 3
    no_improve = 0
    
    print(f"Starting training with RGCL using {config.MODEL_NAME}...")
    print(f"Training parameters: BatchSize={config.BATCH_SIZE}, LR={config.LEARNING_RATE}, Initial Alpha={dynamic_loss.initial_alpha}")
    if index is not None:
        print("SMOTE applied to features used for retrieval index.")
    else:
        print("SMOTE could not be applied or index building failed.")
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss, train_ce_loss, train_cl_loss = 0.0, 0.0, 0.0
        train_preds, train_true = [], []

        for batch_idx, batch in enumerate(train_loader):
            try:
                inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in ["pixel_values", "input_ids", "attention_mask"]
                }
                labels = batch["label"].to(device)

                outputs = model(**inputs)
                logits = outputs["logits"]
                fused_emb = outputs["fused_embeds"]

                # --- Loss Calculation ---
                # 1. Cross-Entropy Loss (在原始批次数据上)
                ce_loss = criterion_ce(logits, labels)

                # 2. Contrastive Loss (使用 SMOTE 增强的索引)
                cl_loss_val = torch.tensor(0.0, device=device)
                if index is not None and norm_features is not None and retrieval_labels is not None:
                    try:
                        # 使用当前批次的融合特征查询 SMOTE 索引
                        query_features = fused_emb.detach().cpu()
                        _, indices = retrieve_similar_samples(
                            index,
                            query_features,
                            k=config.RETRIEVAL_K
                        )

                        # 获取检索到的特征 (来自 SMOTE 处理过的 norm_features) 和标签
                        # 确保 retrieval_labels 在 CPU 上以便索引
                        retrieval_labels_cpu = retrieval_labels.cpu()
                        retrieval_features_batch = []
                        retrieval_labels_batch = []
                        feature_dim = norm_features.size(1) # Should be 768

                        for i, idx_list in enumerate(indices):
                            # 确保索引对于可能更大的 SMOTE 索引有效
                            valid_idx_list = [idx for idx in idx_list if idx >= 0 and idx < len(norm_features)]
                            
                            if not valid_idx_list:
                                # 处理没有有效检索结果的情况 (例如，填充)
                                sample_retrieval_features = torch.zeros((config.RETRIEVAL_K, feature_dim), dtype=norm_features.dtype)
                                sample_retrieval_labels = torch.full((config.RETRIEVAL_K,), -1, dtype=retrieval_labels_cpu.dtype) # 使用 -1 表示无效
                            else:
                                sample_retrieval_features = norm_features[valid_idx_list]
                                sample_retrieval_labels = retrieval_labels_cpu[valid_idx_list]

                                # 如果检索到的/有效的样本少于 K 个，则进行填充
                                if len(valid_idx_list) < config.RETRIEVAL_K:
                                    num_missing = config.RETRIEVAL_K - len(valid_idx_list)
                                    padding_features = torch.zeros((num_missing, feature_dim), dtype=sample_retrieval_features.dtype)
                                    padding_labels = torch.full((num_missing,), -1, dtype=sample_retrieval_labels.dtype)
                                    sample_retrieval_features = torch.cat([sample_retrieval_features, padding_features], dim=0)
                                    sample_retrieval_labels = torch.cat([sample_retrieval_labels, padding_labels], dim=0)

                            retrieval_features_batch.append(sample_retrieval_features)
                            retrieval_labels_batch.append(sample_retrieval_labels)

                        retrieval_features_batch = torch.stack(retrieval_features_batch).to(device)
                        retrieval_labels_batch = torch.stack(retrieval_labels_batch).to(device)

                        cl_loss_val = criterion_cl(
                            fused_emb,              # anchor (batch_size, 768)
                            labels,                 # anchor labels (batch_size,)
                            retrieval_features_batch, # retrieved (batch_size, k, 768 from SMOTE'd index)
                            retrieval_labels_batch  # retrieved labels (batch_size, k from SMOTE'd index)
                        )

                    except Exception as e:
                        print(f"计算对比损失时出错 (Batch {batch_idx}): {str(e)}")
                        cl_loss_val = torch.tensor(0.0, device=device)

                # 3. 使用动态包装器组合损失
                # 将 epoch 编号 (0-indexed) 传递给包装器
                loss = dynamic_loss(ce_loss, cl_loss_val, epoch)

                # --- Backpropagation ---
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * labels.size(0)
                train_ce_loss += ce_loss.item() * labels.size(0)
                # 检查 CL 是否实际计算
                if cl_loss_val.item() > 0: 
                    train_cl_loss += cl_loss_val.item() * labels.size(0)

                train_preds.extend(logits.argmax(1).cpu().numpy())
                train_true.extend(labels.cpu().numpy())

                # Periodic print
                if batch_idx % 20 == 0:
                    current_alpha = dynamic_loss.get_alpha(epoch)
                    print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} Batch {batch_idx}/{len(train_loader)}: "
                        f"Loss={loss.item():.4f} (Alpha={current_alpha:.3f}) CE={ce_loss.item():.4f} CL={cl_loss_val.item():.4f}")


            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {str(e)}")
                continue # Skip batch on error

        # --- End of Epoch ---
        scheduler.step()

        train_acc = accuracy_score(train_true, train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_true, train_preds, average='binary', zero_division=0
        )
        # 评估仅使用 CE 损失
        val_metrics = evaluate_rgcl(model, val_loader, criterion_ce, device) 

        # 使用实际处理的样本数
        num_train_samples = len(train_true) 
        avg_train_loss = train_loss / num_train_samples if num_train_samples > 0 else 0
        avg_train_ce = train_ce_loss / num_train_samples if num_train_samples > 0 else 0
        avg_train_cl = train_cl_loss / num_train_samples if num_train_samples > 0 else 0

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"Train - Loss: {avg_train_loss:.4f} CE: {avg_train_ce:.4f} CL: {avg_train_cl:.4f}")
        print(f"Train - Acc: {train_acc:.4f} F1: {train_f1:.4f} Prec: {train_precision:.4f} Rec: {train_recall:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
              f"F1: {val_metrics['f1']:.4f} Prec: {val_metrics['precision']:.4f} Rec: {val_metrics['recall']:.4f}")

        # Early stopping and best model saving
        current_val_f1 = val_metrics['f1']
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # 更新文件名以包含维度和融合方法
            model_filename = os.path.join(
                config.OUTPUT_DIR,
                f"rgcl_{model_name_slug}_dim{embed_dim}_{fusion_method_slug}_smote_best.pth" 
            )
            try:
                 torch.save(best_state, model_filename)
                 print(f"  最佳模型已保存到: {model_filename}")
            except Exception as e:
                 print(f"  保存最佳模型时出错: {e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs due to no improvement in Val F1.")
                break

    # --- End of Training ---
    # 加载最佳模型，保存它，并在测试集上评估
    if best_state is not None:
        model.load_state_dict(best_state)
        # 更新模型文件名以反映 SMOTE 和维度信息
        model_filename = os.path.join(
            config.OUTPUT_DIR,
            f"rgcl_{model_name_slug}_dim{embed_dim}_{fusion_method_slug}_smote_best.pth" # 加入维度信息
        )
        torch.save(best_state, model_filename)
        print(f"Best model saved to {model_filename}")
    else:
        print("Warning: No best model state found. Testing with the final model state.")


    test_metrics = evaluate_rgcl(model, test_loader, criterion_ce, device)

    print("\n===== Final Test Results (Best Model) =====")
    print(f"Test - Loss: {test_metrics['loss']:.4f} Acc: {test_metrics['accuracy']:.4f} "
          f"F1: {test_metrics['f1']:.4f} Prec: {test_metrics['precision']:.4f} Rec: {test_metrics['recall']:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        test_metrics['true_labels'],
        test_metrics['predictions'],
        target_names=["Not Offensive", "Offensive"],
        zero_division=0
    ))

    print(f"Total training time: {time.time()-start_time:.2f}s")

    return model, test_metrics

def evaluate_rgcl(model, loader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            try:
                inputs = {
                    k: v.to(device) 
                    for k, v in batch.items() 
                    if k in ["pixel_values", "input_ids", "attention_mask"]
                }
                labels = batch["label"].to(device)
                
                outputs = model(**inputs)
                logits = outputs["logits"]
                
                loss = criterion(logits, labels)
                
                total_loss += loss.item() * labels.size(0)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"评估批次时出错: {str(e)}")
                continue
    
    # 确保有预测结果
    if not all_preds:
        return {
            'loss': float('inf'),
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'predictions': [],
            'true_labels': []
        }
        
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    avg_loss = total_loss / len(loader.dataset)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'true_labels': all_labels
    }

if __name__ == "__main__":
    df = load_and_clean_data()
    train_df, val_df, test_df = preprocess_data(df)
    
    processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = CLIPMemotionDataset(
        train_df, 
        Config.IMAGES_DIR, 
        processor, 
        augment=Config.USE_DATA_AUGMENTATION
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=0  # 避免多进程问题
    )
    
    val_dataset = CLIPMemotionDataset(val_df, Config.IMAGES_DIR, processor)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=0)
    
    test_dataset = CLIPMemotionDataset(test_df, Config.IMAGES_DIR, processor)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=0)
    
    model = RGCLCLIPClassifier(
        Config.MODEL_NAME, 
        freeze_clip=Config.FREEZE_CLIP
    ).to(DEVICE)
    
    trained_model, _ = train_rgcl_model(
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        DEVICE, 
        config=Config
    )