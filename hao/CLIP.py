import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPModelWrapper:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        初始化 CLIP 模型和处理器
        :param model_name: 使用的预训练模型名称
        :param device: 使用的设备（'cuda' 或 'cpu'）
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_text(self, texts):
        """
        对输入文本进行编码
        :param texts: 字符串列表
        :return: 文本特征张量
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化

    def encode_image(self, images):
        """
        对输入图像进行编码
        :param images: PIL Image 对象或 PIL 图像列表
        :return: 图像特征张量
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化

    def calculate_similarity(self, image_features, text_features):
        """
        计算图像和文本特征之间的余弦相似度
        :param image_features: 图像特征张量
        :param text_features: 文本特征张量
        :return: 相似度张量
        """
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        return (100.0 * similarity).softmax(dim=-1)

    def classify_image(self, image, categories):
        """
        对图像进行分类
        :param image: 单张 PIL 图像
        :param categories: 类别描述列表
        :return: 最可能的类别和相似度
        """
        text_features = self.encode_text([f"A photo of {category}" for category in categories])
        image_features = self.encode_image([image])
        similarities = self.calculate_similarity(image_features, text_features)
        best_idx = similarities.argmax()
        return categories[best_idx], similarities[best_idx].item()

    def batch_classify_images(self, images, categories):
        """
        对多张图像进行分类
        :param images: PIL 图像列表
        :param categories: 类别描述列表
        :return: 每张图像的分类结果
        """
        text_features = self.encode_text([f"A photo of {category}" for category in categories])
        image_features = self.encode_image(images)
        results = []
        for i, img_feat in enumerate(image_features):
            similarities = self.calculate_similarity(img_feat.unsqueeze(0), text_features)
            best_idx = similarities.argmax()
            results.append((categories[best_idx], similarities[best_idx].item()))
        return results
