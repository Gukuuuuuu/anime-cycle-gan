import torch
from torchvision import transforms


# 配置参数
class Config:
    # 数据集路径
    trainA_path = "trainA"  # 真实人像
    trainB1_path = "trainB1"  # 风格1
    trainB2_path = "trainB2"  # 风格2
    test_path = "testA"  # 测试集

    # 训练参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    lr = 0.0002
    epochs = 50
    img_size = 256

    # 模型保存路径
    model1_save_path = "saved_models/cyclegan_model1"
    model2_save_path = "saved_models/cyclegan_model2"

    # 结果保存路径
    results_path = "results"

config = Config()
# 数据处理管道
transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])