from torch.utils.tensorboard import SummaryWriter

# 创建编辑器，保存日志，指令保存路径log_dir
writer = SummaryWriter(log_dir="./logs_tensorboard")  # 指定保存位置

# y = 2 * x
for i in range(100):
    # 添加标题，x轴，y轴
    # tag: 标题名， scalar_value: y轴， global_step: x轴
    writer.add_scalar(tag="y=2x", scalar_value=2*1, global_step=i)

# 关闭
writer.close()

