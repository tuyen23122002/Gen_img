import os
import time
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import *
from dataset import *
from models import *
import logging
logger_py = logging.getLogger(__name__)

def main(config):
    # Đặt seed ngẫu nhiên và thiết bị thực thi
    set_seed(config.seed)
    # Đặt thiết bị
    device = set_device(config.device)

    # Thư mục đầu ra và các shortcut
    out_dir = config.out_dir
    vis_dir = os.path.join(out_dir, 'vis')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(vis_dir)

    print_every = config.print_every
    checkpoint_every = config.checkpoint_every
    validate_every = config.validate_every
    visualize_every = config.visualize_every
    backup_every = config.backup_every
    exit_after = config.exit_after

    model_selection_metric = config.model_selection_metric
    model_selection_sign = 1 if config.model_selection_mode == 'maximize' else -1

    # Chuẩn bị dữ liệu hoặc dataloader
    train_loader = get_loader_train(config)
    val_loader = get_loader_val(config)
    fixed_data = next(iter(val_loader))
    test_loader = None

    # Chuẩn bị mô hình
    generator = Generator(device, theta=0.8, iteration=4, ae_lambda=[0.6, 0.8, 1.0]).to(device)
    discriminator = Discriminator().to(device)
    vgg16 = Vgg(vgg_init(device))
    model = BlendModel(device, generator=generator, discriminator=discriminator)

    # Khởi tạo huấn luyện
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
    trainer = Trainer(model, vgg16, optimizer_g, optimizer_d, device=device, vis_dir=vis_dir, overwrite_visualization=False)
    checkpoint_io = CheckpointIO(config.out_dir, model=model, optimizer_g=optimizer_g, optimizer_d=optimizer_d)

    try:
        load_dict = checkpoint_io.load('model.pt')
        print("Đã tải checkpoint mô hình.")
    except FileExistsError:
        load_dict = dict()
        print("Không tìm thấy checkpoint mô hình.")

    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get('loss_val_best', -model_selection_sign * np.inf)

    print('Giá trị tốt nhất hiện tại của metric trên tập validation (%s): %.8f' % (model_selection_metric, metric_val_best))

    # Đặt logger và SummaryWriter của TensorBoard
    set_logger(config)
    logger = SummaryWriter(os.path.join(config.out_dir, 'logs'))

    # In mô hình
    nparameters = sum(p.numel() for p in model.parameters())
    logger_py.info(model)
    logger_py.info('Tổng số tham số: %d' % nparameters)

    if hasattr(model, "generator") and model.generator is not None:
        nparameters_g = sum(p.numel() for p in model.generator.parameters())
        logger_py.info('Tổng số tham số của generator: %d' % nparameters_g)

    if hasattr(model, "discriminator") and model.discriminator is not None:
        nparameters_d = sum(p.numel() for p in model.discriminator.parameters())
        logger_py.info('Tổng số tham số của discriminator: %d' % nparameters_d)

    t0 = time.time()
    t0b = time.time()
    while True:
        epoch_it += 1

        for batch in train_loader:
            it += 1
            loss = trainer.train_step(batch, it)
            for (k, v) in loss.items():
                logger.add_scalar(k, v, it)

            if print_every > 0 and (it % print_every) == 0:
                info_txt = '[Epoch %02d] it=%03d, time=%.3f' % (epoch_it, it, time.time() - t0b)
                for (k, v) in loss.items():
                    info_txt += ', %s: %.4f' % (k, v)
                logger_py.info(info_txt)
                t0b = time.time()

            if visualize_every > 0 and (it % visualize_every) == 0:
                logger_py.info('Trực quan hóa')
                image_grid = trainer.visualize(fixed_data, it=it)
                if image_grid is not None:
                    logger.add_image('images', image_grid, it)

            if checkpoint_every > 0 and (it % checkpoint_every) == 0:
                logger_py.info('Đang lưu checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

            if backup_every > 0 and (it % backup_every) == 0:
                logger_py.info('Sao lưu checkpoint')
                checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

            if validate_every > 0 and (it % validate_every) == 0 and (it > 0):
                print("Thực hiện bước đánh giá.")
                eval_dict = trainer.evaluate(val_loader)
                metric_val = eval_dict[model_selection_metric]
                logger_py.info('Metric validation (%s): %.4f' % (model_selection_metric, metric_val))

                for k, v in eval_dict.items():
                    logger.add_scalar('val/%s' % k, v, it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    logger_py.info('Mô hình tốt nhất mới (%s %.4f)' % (model_selection_metric, metric_val_best))
                    checkpoint_io.backup_model_best('model_best.pt')
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

            if exit_after > 0 and (time.time() - t0) >= exit_after:
                logger_py.info('Đã đạt giới hạn thời gian. Thoát.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
                exit(3)

        if epoch_it > config.epoch_size:
            logger_py.info('Số epoch đạt giới hạn. Thoát.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình a3f (absolutely fake feature field model).')
    parser.add_argument('--seed', type=int, default=42, help='Seed ngẫu nhiên')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', type=int, default=-1, help='Số thiết bị, device=-1 dùng CPU')
    parser.add_argument('--model_selection_metric', type=str, default='psnr', choices=['psnr', 'ssim'])
    parser.add_argument('--model_selection_mode', type=str, default='maximize', choices=['maximize', 'minimize'])
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--length', type=int, default=None, help='Chỉ dùng trong debug, tải nhanh dữ liệu nhỏ')
    parser.add_argument('--img_size', type=int, default=256, help='Kích thước hình ảnh')
    parser.add_argument('--epoch_size', type=int, default=100, help='Số lượng epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='Kích thước batch')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--checkpoint_every', type=int, default=500)
    parser.add_argument('--validate_every', type=int, default=10000)
    parser.add_argument('--visualize_every', type=int, default=1000)
    parser.add_argument('--backup_every', type=int, default=1000000)
    parser.add_argument('--exit_after', type=int, default=-1, help='Thoát sau số giây được chỉ định với mã thoát là 2.')

    config = parser.parse_args(
        '''
        --seed 42
        --train_dir D:/Downloads/Project_gen/attentive-gan-derainnet-pytorch/data_root/data_train
        --val_dir D:/Downloads/Project_gen/attentive-gan-derainnet-pytorch/data_root/data_test
        --batch_size 4
        --device 1
        --checkpoint_every 500
        --validate_every 500
        --visualize_every 500
        '''.split()
    )

    if config.debug:
        config.batch_size = 1
        config.length = 10
        config.print_every = 1
        config.checkpoint_every = 1
        config.validate_every = 1
        config.visualize_every = 1
        config.backup_every = 1
        config.img_size = 256
    main(config)
