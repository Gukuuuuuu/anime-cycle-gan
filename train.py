import os
from torch.utils.data import DataLoader
from Config import Config, transform
from Dataset import ImageDataset
from model import CycleGAN

config = Config()

# 创建目录
os.makedirs(config.model1_save_path, exist_ok=True)
os.makedirs(config.model2_save_path, exist_ok=True)
os.makedirs(config.results_path, exist_ok=True)

# 训练模型1：真实人像 <-> 风格1
def train_model1():
    print("开始训练模型1: 真实人像 <-> 风格1")

    # 数据集
    domainA_paths = []
    for filename in os.listdir(config.trainA_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            domainA_paths.append(os.path.join(config.trainA_path, filename))

    domainB_paths = []
    for filename in os.listdir(config.trainB1_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            domainB_paths.append(os.path.join(config.trainB1_path, filename))


    # 创建数据集
    dataset = ImageDataset(domainA_paths, domainB_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    # 初始化模型
    cyclegan = CycleGAN(config.device)

    # 训练循环
    for epoch in range(config.epochs):
        losses = {'loss_G': 0, 'loss_D': 0, 'loss_cycle': 0}
        num_batches = 0

        for i, batch in enumerate(dataloader):
            real_A, real_B = batch
            step_losses = cyclegan.train_step(real_A, real_B)

            # 累积损失
            losses['loss_G'] += step_losses['loss_G']
            losses['loss_D'] += step_losses['loss_D']
            losses['loss_cycle'] += step_losses['loss_cycle']
            num_batches += 1

            # 每50个batch打印一次进度
            if i % 50 == 0:
                print(f"[Epoch {epoch}/{config.epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss G: {step_losses['loss_G']:.4f} Loss D: {step_losses['loss_D']:.4f}")

        # 计算平均损失
        avg_loss_G = losses['loss_G'] / num_batches
        avg_loss_D = losses['loss_D'] / num_batches
        avg_loss_cycle = losses['loss_cycle'] / num_batches

        print(f"[Epoch {epoch}/{config.epochs}] "
              f"Avg Loss G: {avg_loss_G:.4f} "
              f"Avg Loss D: {avg_loss_D:.4f} "
              f"Avg Cycle Loss: {avg_loss_cycle:.4f}")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(config.model1_save_path, f"cyclegan_model1_epoch{epoch + 1}.pth")
            cyclegan.save_model(model_path)
            print(f"保存模型1到 {model_path}")

    print("模型1训练完成")


# 训练模型2：真实人像 <-> 风格2
def train_model2():
    print("开始训练模型2: 真实人像 <-> 风格2")

    # 数据集
    dataset = ImageDataset(config.trainA_path, config.trainB2_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # 初始化模型
    cyclegan = CycleGAN(config.device)

    # 训练循环
    for epoch in range(config.epochs):
        losses = {'loss_G': 0, 'loss_D': 0, 'loss_cycle': 0}
        num_batches = 0

        for i, batch in enumerate(dataloader):
            real_A, real_B = batch
            step_losses = cyclegan.train_step(real_A, real_B)

            # 累积损失
            losses['loss_G'] += step_losses['loss_G']
            losses['loss_D'] += step_losses['loss_D']
            losses['loss_cycle'] += step_losses['loss_cycle']
            num_batches += 1

            # 每50个batch打印一次进度
            if i % 50 == 0:
                print(f"[Epoch {epoch}/{config.epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss G: {step_losses['loss_G']:.4f} Loss D: {step_losses['loss_D']:.4f}")

        # 计算平均损失
        avg_loss_G = losses['loss_G'] / num_batches
        avg_loss_D = losses['loss_D'] / num_batches
        avg_loss_cycle = losses['loss_cycle'] / num_batches

        print(f"[Epoch {epoch}/{config.epochs}] "
              f"Avg Loss G: {avg_loss_G:.4f} "
              f"Avg Loss D: {avg_loss_D:.4f} "
              f"Avg Cycle Loss: {avg_loss_cycle:.4f}")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(config.model2_save_path, f"cyclegan_model2_epoch{epoch + 1}.pth")
            cyclegan.save_model(model_path)
            print(f"保存模型2到 {model_path}")

    print("模型2训练完成")





if __name__ == "__main__":
    # 步骤1: 训练模型1（真实人像 <-> 风格1）
    train_model1()

    # 步骤2: 训练模型2（真实人像 <-> 风格2）
    train_model2()


    print("所有任务已完成！")