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
    MODEL_NAME = "openai/clip-vit-base-patch32"  # 可替换为其他CLIP模型
    FREEZE_CLIP = False  # 是否冻结CLIP模型权重
    
    # 训练配置
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    TEMPERATURE = 0.07  # 对比损失温度参数
    
    # 检索配置
    RETRIEVAL_K = 5  # 每个样本检索的相似样本数量
    ALPHA = 0.7  # 交叉熵损失和对比损失的权重比例
    
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
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
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
                
                # 获取CLIP输出的特征
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
            
    # 将所有特征堆叠为单个张量
    features = torch.cat(features_list, dim=0)
    img_features = torch.cat(img_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
    labels = torch.tensor(labels_list)
    
    # 构建结果字典
    result = {
        "features": features,
        "img_features": img_features,
        "text_features": text_features,
        "labels": labels,
        "image_names": image_names
    }
    
    # 缓存结果
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
    
    # 获取特征维度
    dim = features.shape[1]
    
    # 使用内积作为相似度度量（余弦相似度）
    index = faiss.IndexFlatIP(dim)
    
    # 归一化特征进行余弦相似度计算
    norm_features = F.normalize(features, p=2, dim=1)
    
    # 确保特征是float32类型，这是FAISS要求的
    norm_features_np = norm_features.numpy().astype(np.float32)
    
    # 添加到索引
    index.add(norm_features_np)
    
    return index, norm_features

# 检索相似样本
def retrieve_similar_samples(index, query_features, k=5, exclude_self=True):
    """检索与查询特征最相似的k个样本"""
    # 归一化查询特征
    norm_query = F.normalize(query_features, p=2, dim=1)
    
    # 确保查询特征是float32类型
    norm_query_np = norm_query.numpy().astype(np.float32)
    
    # 如果要排除自身，需要多检索一个结果
    actual_k = k + 1 if exclude_self else k
    
    # 执行检索
    scores, indices = index.search(norm_query_np, actual_k)
    
    # 排除第一个结果（自身）
    if exclude_self:
        indices = indices[:, 1:]
        scores = scores[:, 1:]
    
    return scores, indices

# 修复后的SimpleContrastiveLoss类实现 - 解决mat1 and mat2 shapes cannot be multiplied错误

class SimpleContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, retrieval_features, retrieval_labels):
        """
        计算检索增强的对比损失

        参数:
            features: 形状为(batch_size, feature_dim)的特征 (锚点, 512维)
            labels: 形状为(batch_size,)的标签
            retrieval_features: 形状为(batch_size, k, feature_dim)的检索特征 (512维)
            retrieval_labels: 形状为(batch_size, k)的检索标签
        """
        batch_size = features.size(0)
        feature_dim = features.size(1) # 应该是512
        k = retrieval_features.size(1)

        # 验证输入维度 (可选的健全性检查)
        if retrieval_features.size(2) != feature_dim:
             print(f"警告: SimpleContrastiveLoss 中特征维度不匹配! Anchor: {feature_dim}, Retrieved: {retrieval_features.size(2)}")
             # 可以选择抛出错误或尝试继续，但根本问题应在数据准备阶段解决
             # return torch.tensor(0.0, device=features.device) # 或者返回0损失

        # 归一化锚点特征 (batch_size, feature_dim)
        features_norm = F.normalize(features, p=2, dim=1)

        total_loss = 0.0
        valid_samples = 0 # 计数实际计算了损失的样本

        for i in range(batch_size):
            # 当前样本特征 (1, feature_dim)
            anchor_norm = features_norm[i].unsqueeze(0)
            anchor_label = labels[i]

            # 当前样本的检索结果 (k, feature_dim) 和标签 (k,)
            retrieved_norm = F.normalize(retrieval_features[i], p=2, dim=1)
            retrieved_labels = retrieval_labels[i]

            # 计算相似度: (1, feature_dim) x (feature_dim, k) -> (1, k)
            similarity = torch.mm(anchor_norm, retrieved_norm.t()) / self.temperature
            similarity = similarity.squeeze(0) # 展平为 (k,)

            # 创建标签掩码 (同类样本为正样本), 忽略填充的无效标签 (-1)
            positive_mask = (retrieved_labels == anchor_label).float()
            valid_retrieval_mask = (retrieved_labels != -1).float() # 掩码掉填充的样本

            # 如果没有有效的正样本，则跳过此样本的损失计算
            # (确保我们只考虑有效的检索结果)
            if (positive_mask * valid_retrieval_mask).sum() == 0:
                continue

            # 计算对比损失(InfoNCE)
            exp_sim = torch.exp(similarity)

            # 只考虑有效的检索结果计算损失
            exp_sim_valid = exp_sim * valid_retrieval_mask

            pos_sum = torch.sum(exp_sim_valid * positive_mask) + 1e-10 # 正样本的和
            all_sum = torch.sum(exp_sim_valid) + 1e-10 # 所有有效负样本+正样本的和

            # 计算单个样本的损失
            sample_loss = -torch.log(pos_sum / all_sum)

            # 累加损失，仅当损失有效时
            if not torch.isnan(sample_loss) and not torch.isinf(sample_loss):
                total_loss += sample_loss
                valid_samples += 1

        # 平均损失，只基于有效计算的样本
        return total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0, device=features.device)

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
    def __init__(self, model_name, num_labels=2, freeze_clip=False):
        super().__init__()
        # 加载CLIP模型
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.embed_dim = self.clip_model.config.projection_dim
        
        # 是否冻结CLIP模型参数
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # 特征融合层
        self.fusion = self._create_fusion_layer()
        
        # 分类器层
        self.classifier = self._create_classifier(num_labels)
    
    def _create_fusion_layer(self):
        """创建多模态特征融合层"""
        return nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def _create_classifier(self, num_labels):
        """创建分类器层"""
        return nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, pixel_values, input_ids, attention_mask):
        """前向传播，返回多个中间结果以用于不同学习任务"""
        # 运行CLIP模型获取图像和文本嵌入
        outputs = self.clip_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # 获取图像和文本的表示
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # 融合图像和文本特征
        concat_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        fused_embeds = self.fusion(concat_embeds)
        
        # 分类
        logits = self.classifier(fused_embeds)
        
        # 返回多个输出以便于不同的学习任务
        return {
            "logits": logits,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "fused_embeds": fused_embeds
        }

# 新增：提取用于构建索引的融合特征
def extract_fused_features_for_index(model, dataframe, images_dir, processor, device, batch_size=32, cache_file=None):
    """使用模型提取融合特征以构建FAISS索引"""
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached fused embeddings from {cache_file}")
        return torch.load(cache_file)

    print("Extracting fused features for FAISS index...")
    model.eval()  # 设置为评估模式

    # 创建一个简单的数据加载器
    dataset = CLIPMemotionDataset(dataframe, images_dir, processor, augment=False) # 不使用增强
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

                # 通过模型获取输出
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

    # 合并结果
    features = torch.cat(all_fused_embeds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    result = {
        "features": features, # 现在这是512维的融合特征
        "labels": labels,
        "image_names": all_image_names
    }

    # 缓存结果
    if cache_file:
        torch.save(result, cache_file)
        print(f"Saved fused embeddings to {cache_file}")

    model.train() # 恢复训练模式
    return result

# === SMOTE Integration Start ===
# SMOTE特征增强核心代码
def apply_smote(features, labels):
    """
    Applies SMOTE to features after PCA dimensionality reduction.
    
    Args:
        features (torch.Tensor or np.ndarray): High-dimensional features.
        labels (torch.Tensor or np.ndarray): Corresponding labels.
        
    Returns:
        tuple: (resampled_features_original_dim, resampled_labels) as numpy arrays.
    """
    print(f"Original training data shape: {features.shape}, Label distribution: {np.bincount(labels)}")
    
    # Ensure data is on CPU and in numpy format
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    else:
        features_np = features
        
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels

    # Apply PCA
    # Using min(n_samples, n_features) for n_components if features < 32
    n_components_pca = min(min(features_np.shape), 32)
    if n_components_pca < 2: # SMOTE needs at least 2 dimensions
        print(f"Warning: Not enough features ({features_np.shape[1]}) or samples ({features_np.shape[0]}) for PCA + SMOTE. Skipping SMOTE.")
        return features_np, labels_np

    print(f"Applying PCA with n_components={n_components_pca}")
    pca = PCA(n_components=n_components_pca)
    try:
        low_dim = pca.fit_transform(features_np)
    except Exception as e:
        print(f"Error during PCA: {e}. Skipping SMOTE.")
        return features_np, labels_np

    # Apply SMOTE
    # k_neighbors should be less than the number of minority samples
    minority_count = np.sum(labels_np == 1) # Assuming 1 is the minority class
    k = min(5, minority_count - 1) if minority_count > 1 else 1
    
    if k < 1:
         print(f"Warning: Not enough minority samples ({minority_count}) for SMOTE. Skipping SMOTE.")
         return features_np, labels_np
         
    print(f"Applying SMOTE with k_neighbors={k}")
    try:
        sm = SMOTE(k_neighbors=k, random_state=42)
        resampled_low_dim, res_labels = sm.fit_resample(low_dim, labels_np)
    except Exception as e:
        print(f"Error during SMOTE: {e}. Skipping SMOTE.")
        return features_np, labels_np

    print(f"Resampled data shape: {resampled_low_dim.shape}, Resampled label distribution: {np.bincount(res_labels)}")

    # Add Gaussian noise to synthetic samples (optional, helps prevent overfitting)
    # Identify synthetic samples added by SMOTE
    num_original = len(features_np)
    if len(resampled_low_dim) > num_original:
        noise = np.random.normal(0, 0.05, resampled_low_dim[num_original:].shape) # Smaller noise
        resampled_low_dim[num_original:] += noise
        print(f"Added noise to {len(resampled_low_dim) - num_original} synthetic samples.")

    # Inverse transform back to original dimension
    try:
        resampled_original_dim = pca.inverse_transform(resampled_low_dim)
    except Exception as e:
        print(f"Error during PCA inverse transform: {e}. Returning low-dimensional resampled data.")
        # Fallback: return low-dim data if inverse transform fails
        return resampled_low_dim, res_labels 

    print("SMOTE application finished.")
    return resampled_original_dim, res_labels

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
        # Ensure cl_loss is a tensor
        if not isinstance(cl_loss, torch.Tensor):
            cl_loss = torch.tensor(cl_loss, device=ce_loss.device) # Make sure device matches
            
        combined_loss = current_alpha * ce_loss + (1.0 - current_alpha) * cl_loss
        # print(f"Epoch {epoch+1}: Alpha={current_alpha:.4f}, CE={ce_loss.item():.4f}, CL={cl_loss.item():.4f}, Combined={combined_loss.item():.4f}") # Debug print
        return combined_loss
# === SMOTE Integration End ===

# 训练函数 - 使用检索增强的对比学习
def train_rgcl_model(model, train_loader, val_loader, test_loader, device, config=None):
    """训练RGCL模型函数"""
    if config is None:
        config = Config
    
    # 优化器设置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.NUM_EPOCHS
    )
    
    # 计算类别权重
    train_labels_list = [item for item in train_df["label"]] # Use original train_df labels
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.array([0, 1]), 
        y=train_labels_list
    )
    
    # 损失函数
    criterion_ce = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device)
    )
    criterion_cl = SimpleContrastiveLoss(temperature=config.TEMPERATURE)
    # === Dynamic Loss Wrapper ===
    dynamic_loss = DynamicLossWrapper(
        initial_alpha=config.ALPHA, # Use config.ALPHA as starting point
        final_alpha=0.95,           # Gradually increase focus on CE loss
        total_epochs=config.NUM_EPOCHS
    )

    # === Feature Extraction & SMOTE Application ===
    cache_filename = f"train_fused_features_{config.MODEL_NAME.split('/')[-1]}.pt"
    cache_file = os.path.join(config.EMBEDDING_CACHE_DIR, cache_filename)

    index = None
    norm_features = None
    retrieval_labels = None # Labels corresponding to the features in the index

    try:
        # 1. Extract original fused features
        train_fused_features_data = extract_fused_features_for_index(
            model,
            train_df,
            Config.IMAGES_DIR,
            train_loader.dataset.processor,
            device,
            batch_size=config.BATCH_SIZE,
            cache_file=cache_file # Use cache if available
        )
        original_features = train_fused_features_data["features"] # Tensor
        original_labels = train_fused_features_data["labels"]     # Tensor

        # 2. Apply SMOTE (operates on numpy arrays)
        # Pass tensors directly, apply_smote handles conversion
        smote_features_np, smote_labels_np = apply_smote(
            original_features,
            original_labels
        )

        # Convert back to tensors for FAISS
        smote_features = torch.from_numpy(smote_features_np).float()
        smote_labels = torch.from_numpy(smote_labels_np).long()

        # 3. Build FAISS index using SMOTE-augmented features
        print("Building FAISS index with SMOTE-augmented features...")
        # build_faiss_index expects a tensor
        index, norm_features = build_faiss_index(smote_features)
        retrieval_labels = smote_labels # Use SMOTE labels for retrieval check
        print(f"FAISS index built with {norm_features.shape[0]} samples (after SMOTE), dimension {norm_features.shape[1]}.")
        print(f"Label distribution in index: {np.bincount(retrieval_labels.numpy())}")


    except Exception as e:
        print(f"构建索引或应用SMOTE时出错: {str(e)}")
        print("将退回到不使用检索增强和SMOTE的训练方式")
        index = None
        norm_features = None
        retrieval_labels = None
        # Fallback: Use original features if SMOTE/Indexing fails
        if 'original_features' in locals():
             try:
                 print("Falling back to building index with original features...")
                 index, norm_features = build_faiss_index(original_features)
                 retrieval_labels = original_labels
                 print(f"FAISS index built with {norm_features.shape[0]} original samples.")
             except Exception as e_fallback:
                 print(f"Fallback index building failed: {e_fallback}")
                 index = None # Ensure index is None if fallback fails too
                 norm_features = None
                 retrieval_labels = None


    # === Training Loop Start ===
    # 训练状态追踪
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
                # 准备输入数据
                inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in ["pixel_values", "input_ids", "attention_mask"]
                }
                labels = batch["label"].to(device)

                # 前向传播
                outputs = model(**inputs)
                logits = outputs["logits"]
                fused_emb = outputs["fused_embeds"]

                # --- Loss Calculation ---
                # 1. Cross-Entropy Loss (on original batch data)
                ce_loss = criterion_ce(logits, labels)

                # 2. Contrastive Loss (using SMOTE-augmented index)
                cl_loss_val = torch.tensor(0.0, device=device)
                if index is not None and norm_features is not None and retrieval_labels is not None:
                    try:
                        # Retrieve using current batch's fused features against SMOTE'd index
                        query_features = fused_emb.detach().cpu()
                        _, indices = retrieve_similar_samples(
                            index,
                            query_features,
                            k=config.RETRIEVAL_K
                        )

                        # Get retrieved features (from SMOTE'd norm_features) and labels
                        # Ensure retrieval_labels is on CPU for indexing
                        retrieval_labels_cpu = retrieval_labels.cpu()
                        retrieval_features_batch = []
                        retrieval_labels_batch = []
                        feature_dim = norm_features.size(1) # Should be 512

                        for i, idx_list in enumerate(indices):
                            # Ensure indices are valid for the potentially larger SMOTE'd index
                            valid_idx_list = [idx for idx in idx_list if idx >= 0 and idx < len(norm_features)]
                            
                            if not valid_idx_list:
                                # Handle cases with no valid retrievals (e.g., padding)
                                sample_retrieval_features = torch.zeros((config.RETRIEVAL_K, feature_dim), dtype=norm_features.dtype)
                                sample_retrieval_labels = torch.full((config.RETRIEVAL_K,), -1, dtype=retrieval_labels_cpu.dtype) # Use -1 for invalid
                            else:
                                # Retrieve features and labels using valid indices
                                sample_retrieval_features = norm_features[valid_idx_list]
                                sample_retrieval_labels = retrieval_labels_cpu[valid_idx_list]

                                # Pad if fewer than K samples were retrieved/valid
                                if len(valid_idx_list) < config.RETRIEVAL_K:
                                    num_missing = config.RETRIEVAL_K - len(valid_idx_list)
                                    padding_features = torch.zeros((num_missing, feature_dim), dtype=sample_retrieval_features.dtype)
                                    padding_labels = torch.full((num_missing,), -1, dtype=sample_retrieval_labels.dtype)
                                    sample_retrieval_features = torch.cat([sample_retrieval_features, padding_features], dim=0)
                                    sample_retrieval_labels = torch.cat([sample_retrieval_labels, padding_labels], dim=0)

                            retrieval_features_batch.append(sample_retrieval_features)
                            retrieval_labels_batch.append(sample_retrieval_labels)

                        # Stack and move to device
                        retrieval_features_batch = torch.stack(retrieval_features_batch).to(device)
                        retrieval_labels_batch = torch.stack(retrieval_labels_batch).to(device)

                        # Calculate Contrastive Loss
                        cl_loss_val = criterion_cl(
                            fused_emb,              # anchor (batch_size, 512)
                            labels,                 # anchor labels (batch_size,)
                            retrieval_features_batch, # retrieved (batch_size, k, 512 from SMOTE'd index)
                            retrieval_labels_batch  # retrieved labels (batch_size, k from SMOTE'd index)
                        )

                    except Exception as e:
                        print(f"计算对比损失时出错 (Batch {batch_idx}): {str(e)}")
                        cl_loss_val = torch.tensor(0.0, device=device)

                # 3. Combine Losses using Dynamic Wrapper
                # Pass epoch number (0-indexed) to the wrapper
                loss = dynamic_loss(ce_loss, cl_loss_val, epoch)

                # --- Backpropagation ---
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录损失和预测
                train_loss += loss.item() * labels.size(0)
                train_ce_loss += ce_loss.item() * labels.size(0)
                if cl_loss_val.item() > 0: # Check if CL was actually computed
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

        # Calculate and print epoch metrics
        train_acc = accuracy_score(train_true, train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_true, train_preds, average='binary', zero_division=0
        )
        val_metrics = evaluate_rgcl(model, val_loader, criterion_ce, device) # Evaluation uses only CE loss

        num_train_samples = len(train_true) # Use actual number of processed samples
        avg_train_loss = train_loss / num_train_samples if num_train_samples > 0 else 0
        avg_train_ce = train_ce_loss / num_train_samples if num_train_samples > 0 else 0
        avg_train_cl = train_cl_loss / num_train_samples if num_train_samples > 0 else 0

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"Train - Loss: {avg_train_loss:.4f} CE: {avg_train_ce:.4f} CL: {avg_train_cl:.4f}")
        print(f"Train - Acc: {train_acc:.4f} F1: {train_f1:.4f} Prec: {train_precision:.4f} Rec: {train_recall:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
              f"F1: {val_metrics['f1']:.4f} Prec: {val_metrics['precision']:.4f} Rec: {val_metrics['recall']:.4f}")


        # Early stopping and best model saving
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            print(f"New best model saved! Val F1: {val_metrics['f1']:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs due to no improvement in Val F1.")
                break


    # --- End of Training ---
    # Load best model, save it, and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
        # Update model filename to reflect SMOTE usage potentially
        model_filename = os.path.join(
            config.OUTPUT_DIR,
            f"rgcl_{config.MODEL_NAME.split('/')[-1]}_fused_smote_model.pth" # Indicate SMOTE
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
        
    # 计算性能指标
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
    # 数据加载和预处理
    df = load_and_clean_data()
    train_df, val_df, test_df = preprocess_data(df)
    
    # 初始化CLIP处理器
    processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)
    
    # 创建数据集和数据加载器
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
        num_workers=0  # 避免多进程引起的问题
    )
    
    val_dataset = CLIPMemotionDataset(val_df, Config.IMAGES_DIR, processor)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=0)
    
    test_dataset = CLIPMemotionDataset(test_df, Config.IMAGES_DIR, processor)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=0)
    
    # 初始化RGCL模型
    model = RGCLCLIPClassifier(
        Config.MODEL_NAME, 
        freeze_clip=Config.FREEZE_CLIP
    ).to(DEVICE)
    
    # 训练模型
    trained_model, _ = train_rgcl_model(
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        DEVICE, 
        config=Config
    )