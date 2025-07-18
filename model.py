import torch
import torch.nn as nn
import torch.optim as optim
from Config import Config

config = Config()
# 生成器（U-Net结构）
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 编码器部分 (下采样)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 残差块 (在最低分辨率特征图上应用)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # 解码器部分 (上采样)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # 编码过程 (下采样)
        e1 = self.enc1(x)  # [batch, 64, 128, 128]
        e2 = self.enc2(e1)  # [batch, 128, 64, 64]
        e3 = self.enc3(e2)  # [batch, 256, 32, 32]
        e4 = self.enc4(e3)  # [batch, 512, 16, 16]

        # 残差块处理
        r = self.residual_blocks(e4)  # [batch, 512, 16, 16]

        # 解码过程 (上采样) 带有跳跃连接
        d1 = self.dec1(r)  # [batch, 256, 32, 32]
        # 跳跃连接：连接e3(编码器第3层输出)
        d1 = torch.cat([d1, e3], dim=1)  # [batch, 256+256=512, 32, 32]

        d2 = self.dec2(d1)  # [batch, 128, 64, 64]
        # 跳跃连接：连接e2(编码器第2层输出)
        d2 = torch.cat([d2, e2], dim=1)  # [batch, 128+128=256, 64, 64]

        d3 = self.dec3(d2)  # [batch, 64, 128, 128]
        # 跳跃连接：连接e1(编码器第1层输出)
        d3 = torch.cat([d3, e1], dim=1)  # [batch, 64+64=128, 128, 128]

        # 最终输出层
        d4 = self.dec4(d3)  # [batch, 3, 256, 256]

        return d4


# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)  # 残差连接
# 判别器（PatchGAN）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


# CycleGAN模型封装
class CycleGAN:
    def __init__(self, device):
        self.device = device

        # 初始化生成器和判别器
        self.G_A2B = Generator().to(device)
        self.G_B2A = Generator().to(device)
        self.D_A = Discriminator().to(device)
        self.D_B = Discriminator().to(device)

        # 初始化优化器
        self.optimizer_G = optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
            lr=config.lr, betas=(0.5, 0.999)
        )

        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=config.lr, betas=(0.5, 0.999))

        # 损失函数
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

    def train_step(self, real_A, real_B):
        # 将数据移到设备
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        # 真假标签
        valid = torch.ones((real_A.size(0), 1, 30, 30), requires_grad=False).to(self.device)
        fake = torch.zeros((real_A.size(0), 1, 30, 30), requires_grad=False).to(self.device)

        # ============ 训练生成器 ============
        self.optimizer_G.zero_grad()

        # 身份损失
        identity_B = self.G_A2B(real_B)
        loss_identity_B = self.criterion_identity(identity_B, real_B)

        identity_A = self.G_B2A(real_A)
        loss_identity_A = self.criterion_identity(identity_A, real_A)

        # GAN损失
        fake_B = self.G_A2B(real_A)
        loss_GAN_A2B = self.criterion_gan(self.D_B(fake_B), valid)

        fake_A = self.G_B2A(real_B)
        loss_GAN_B2A = self.criterion_gan(self.D_A(fake_A), valid)

        # 循环一致性损失
        reconstructed_A = self.G_B2A(fake_B)
        loss_cycle_A = self.criterion_cycle(reconstructed_A, real_A)

        reconstructed_B = self.G_A2B(fake_A)
        loss_cycle_B = self.criterion_cycle(reconstructed_B, real_B)

        # 总生成器损失
        loss_G = (
                loss_GAN_A2B + loss_GAN_B2A +
                loss_cycle_A * 10.0 + loss_cycle_B * 10.0 +
                loss_identity_A * 0.5 + loss_identity_B * 0.5
        )

        loss_G.backward()
        self.optimizer_G.step()

        # ============ 训练判别器 A ============
        self.optimizer_D_A.zero_grad()

        # 真实图像损失
        loss_real = self.criterion_gan(self.D_A(real_A), valid)
        # 假图像损失
        loss_fake = self.criterion_gan(self.D_A(fake_A.detach()), fake)

        # 总判别器A损失
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        self.optimizer_D_A.step()

        # ============ 训练判别器 B ============
        self.optimizer_D_B.zero_grad()

        # 真实图像损失
        loss_real = self.criterion_gan(self.D_B(real_B), valid)
        # 假图像损失
        loss_fake = self.criterion_gan(self.D_B(fake_B.detach()), fake)

        # 总判别器B损失
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        self.optimizer_D_B.step()

        return {
            'loss_G': loss_G.item(),
            'loss_D': (loss_D_A.item() + loss_D_B.item()),
            'loss_cycle': (loss_cycle_A.item() + loss_cycle_B.item())
        }

    def save_model(self, path):
        torch.save({
            'G_A2B': self.G_A2B.state_dict(),
            'G_B2A': self.G_B2A.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D_A': self.optimizer_D_A.state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.G_A2B.load_state_dict(checkpoint['G_A2B'])
        self.G_B2A.load_state_dict(checkpoint['G_B2A'])
        self.D_A.load_state_dict(checkpoint['D_A'])
        self.D_B.load_state_dict(checkpoint['D_B'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
        self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
