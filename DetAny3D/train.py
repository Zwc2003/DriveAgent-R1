from detect_anything.datasets.detany3d_dataset import *
from train_utils import *
from wrap_model import WrapModel
from PIL import Image
import cv2
import yaml
from box import Box
from tqdm import tqdm
import os
import argparse
import open3d as o3d
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import datetime
from box import Box
from tqdm import tqdm
import os
from torchvision import transforms
import argparse
from torch.nn.utils.rnn import pad_sequence
from detect_anything.modeling.depth_predictor.unidepth_utils import generate_rays
from torch.utils.tensorboard import SummaryWriter
import random
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext  # Python 3.7+ 提供的空上下文管理器
from torch.distributed import all_gather_object

def parse_args():
    parser = argparse.ArgumentParser(description='set config path') 
    parser.add_argument('--config_path', default='/cpfs01/user/jianghaoran/detany3d/DetAny3D0827/segment_anything/configs/dlc_config.yaml', type=str, help='abosulute path of the config') 
    parser.add_argument('--resume', type=str, help='Path to resume checkpoint')
    parser.add_argument('--exp_dir', type=str, help='Path to save checkpoint')
    args = parser.parse_args() 
    print(f'args: {args}')
    return args

def train_one_epoch(
    cfg,
    model,
    device_id,
    optimizer,
    scheduler,
    train_dataloader,
    epoch,
    logger,
):

    train_dataloader.sampler.set_epoch(epoch)
    model.train()
    loss_epoch = torch.zeros(1).to(device_id)
    loss_dict_epoch = {}
    
    with tqdm(total=len(train_dataloader)) as t:
        for iter, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            with autocast(cfg.use_amp):
                input_dict = {}
                input_dict['images'] = data['images']
                input_dict['images_shape'] = data['before_pad_size']
                input_dict['vit_pad_size'] = data['vit_pad_size']
                image_h, image_w = int(data['before_pad_size'][0, 0]), int(data['before_pad_size'][0, 1])

                # fetch the gt data here
                # bug exists if batch size is not 1.
                depth_gt = data['depth'][:, :image_h, :image_w].to(device_id)
                masks = data['masks'][:, :image_h, :image_w].to(device_id)
                gt_angles = generate_rays(data['K'].to(device_id), (image_h, image_w))[1]
                phi_gt, theta_gt = gt_angles[..., 0], gt_angles[..., 1]

                input_dict["image_for_dino"] = data["image_for_dino"].to(device_id)

                if len(data['prepare_for_dsam']) > 0:
                    # fetch gt here
                    gt_bboxes_2d = torch.stack([data['prepare_for_dsam'][i]['bbox_2d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    gt_center_2d = torch.stack([data['prepare_for_dsam'][i]['center_2d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    gt_bboxes_3d = torch.stack([data['prepare_for_dsam'][i]['bbox_3d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])

                    if cfg.input_point_prompt:
                        input_dict['point_coords'] = torch.stack([data['prepare_for_dsam'][i]['point_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                    if cfg.input_box_prompt:
                        input_dict['boxes_coords'] = torch.stack([data['prepare_for_dsam'][i]['boxes_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])

                    if cfg.input_point_prompt and cfg.input_box_prompt and cfg.random_drop_prompt:
                        if random.random() < cfg.random_drop_prompt_prob:
                            # key_to_drop = random.choice(['point_coords', 'boxes_coords'])
                            del input_dict['boxes_coords']
                        else:
                            del input_dict['point_coords']
                    
                    gt_pose = None
                    if cfg.output_rotation_matrix:
                        gt_pose = torch.stack([data['prepare_for_dsam'][i]['rotation_pose'].to(device_id) for i in range(len(data['prepare_for_dsam']))])

                    # encode gts
                    prompt_num = gt_bboxes_2d.shape[0]
                    encoded_gt_bboxes_2d = gt_bboxes_2d / cfg.model.pad
                    encoded_gt_center_2d = gt_center_2d / cfg.model.pad

                    gt_alpha = -torch.atan2(
                        box_xyxy_to_cxcywh(gt_bboxes_2d)[..., 0] - data['K'][..., 0, 2].to(device_id), data['K'][..., 0, 0].to(device_id)) + gt_bboxes_3d[..., 6]
                    gt_alpha[gt_alpha > torch.pi] = gt_alpha[gt_alpha > torch.pi] - 2 * torch.pi
                    gt_alpha[gt_alpha < -torch.pi] = gt_alpha[gt_alpha < -torch.pi] + 2 * torch.pi

                    gt_angle_cls, gt_angle_res = angle2class(gt_alpha)
                    gt_dir_cls_onehot = F.one_hot(gt_angle_cls.long(), 12).float()

                    encoded_gt_bbox_3d_dim = gt_bboxes_3d[..., 3:6].log()
                    encoded_gt_bbox_3d_depth = gt_bboxes_3d[..., 2:3].log()

                    if cfg.model.multi_level_box_output > 1:
                        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
                        input_dict['gt_angle_cls'] = gt_angle_cls

                only_2d_mode = len(data['prepare_for_dsam']) > 0 and torch.all(gt_bboxes_3d == -1).item()
                if cfg.provide_gt_intrinsics and not only_2d_mode:
                    input_dict['gt_intrinsic'] = data['K']
                
                ret_dict = model(input_dict)
                K_for_convert = ret_dict.get('gt_intrinsic', ret_dict['pred_K']).to(device_id)
                loss_dict = {}
            
                
                pred_angles = generate_rays(ret_dict['pred_K'], (image_h, image_w))[1]
                phi_pred, theta_pred = pred_angles[..., 0], pred_angles[..., 1]

                loss_phi = SILogLoss(phi_pred, phi_gt, coefficient=cfg.loss.phi.coefficient)
                loss_theta = SILogLoss(theta_pred, theta_gt, coefficient=cfg.loss.theta.coefficient)
                loss_depth = SILogLoss(ret_dict['depth_maps'], depth_gt, coefficient=cfg.loss.depth.coefficient, masks=masks, log_mode=True)
            
                if 'depth_loss' in cfg.loss.loss_list and loss_depth is not None:
                    loss_dict['depth_loss'] = cfg.loss.depth.depth_loss_weight * loss_depth
                if 'intrinsic_loss' in cfg.loss.loss_list and not only_2d_mode:
                    loss_dict['loss_phi'] = cfg.loss.phi.phi_loss_weight * loss_phi
                    loss_dict['loss_theta'] = cfg.loss.theta.theta_loss_weight * loss_theta

                if len(data['prepare_for_dsam']) > 0:

                    decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K_for_convert)
                    
                    # 2d giou loss
                    bboxes_pred_2d = ret_dict['pred_bbox_2d']
                    batch_iou_2d = bbox_overlaps_giou(box_cxcywh_to_xyxy(bboxes_pred_2d), encoded_gt_bboxes_2d)
                    loss_box_giou = (1 - batch_iou_2d)

                    # 2d bbox l1 loss
                    loss_l1_bbox = F.smooth_l1_loss(bboxes_pred_2d, box_xyxy_to_cxcywh(encoded_gt_bboxes_2d), reduction='none', beta=1.0 / 9.0)

                    # 3d bbox center 2d loss
                    pred_bbox_3d_center_2d = ret_dict['pred_center_2d']
                    bbox_3d_center_2d_loss = F.smooth_l1_loss(pred_bbox_3d_center_2d, encoded_gt_center_2d, reduction='none', beta=1.0 / 9.0)

                    # 3d bbox depth loss
                    pred_bbox_3d_depth = ret_dict['pred_bbox_3d_depth']
                    depth_log_variance = ret_dict['pred_bbox_3d_depth_log_variance']
                    bbox_3d_depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(pred_bbox_3d_depth - encoded_gt_bbox_3d_depth) + depth_log_variance

                    # 3d bbox dim loss
                    pred_bbox_3d_dim = ret_dict['pred_bbox_3d_dims']
                    bbox_3d_dim_loss = F.smooth_l1_loss(pred_bbox_3d_dim, encoded_gt_bbox_3d_dim, reduction='none', beta=1.0 / 9.0 )

                    bbox_3d_alpha_cls_loss = F.cross_entropy(ret_dict['pred_bbox_3d_alpha_cls'], gt_angle_cls, reduction='none')
                    pred_boxes_3d_alpha_res = torch.sum(ret_dict['pred_bbox_3d_alpha_res'] * gt_dir_cls_onehot, 1)
                    bbox_3d_alpha_res_loss = F.smooth_l1_loss(pred_boxes_3d_alpha_res, gt_angle_res, reduction='none', beta=1.0 / 9.0)
                    bbox_3d_alpha_loss = bbox_3d_alpha_cls_loss + bbox_3d_alpha_res_loss

                    if '2d_bbox_loss' in cfg.loss.loss_list:
                        loss_dict['loss_2d_bbox'] = 5 * loss_l1_bbox.sum() / prompt_num
                        loss_dict['loss_2d_giou'] = 2 * loss_box_giou.sum() / prompt_num
                    if '3d_bbox_loss' in cfg.loss.loss_list and not only_2d_mode:
                        loss_dict['loss_3d_center_2d'] = 10 * bbox_3d_center_2d_loss.sum() / prompt_num
                        loss_dict['loss_3d_depth'] = 1 * bbox_3d_depth_loss.sum() / prompt_num
                        loss_dict['loss_3d_dim'] = 1 * bbox_3d_dim_loss.sum() / prompt_num
                        loss_dict['loss_3d_alpha'] = 1 * bbox_3d_alpha_loss.sum() / prompt_num
                    if cfg.model.multi_level_box_output > 1 and not only_2d_mode and ret_dict.get('box_iou_loss', None) is not None:
                        loss_dict['box_iou_loss'] = 2 * ret_dict['box_iou_loss']
                    
                    pred_pose = None
                    if cfg.output_rotation_matrix:
                        pred_pose = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])

                    decoded_bboxes_pred_3d_corners = compute_3d_bbox_vertices_batch(gt_bboxes_3d, pred_pose)
                    decoded_bboxes_gt_3d_corners = compute_3d_bbox_vertices_batch(gt_bboxes_3d, gt_pose)
                    
                    if 'chamfer_loss' in cfg.loss.loss_list and not only_2d_mode:

                        corners_chamfer_loss = chamfer_loss(decoded_bboxes_pred_3d_corners, decoded_bboxes_gt_3d_corners)
                        loss_dict['corners_chamfer_loss'] = corners_chamfer_loss.sum() / prompt_num

            loss_total = sum([loss_dict[k] for k in loss_dict.keys()])
            loss_epoch += loss_total.detach()
    
            for key, value in loss_dict.items():
                if key not in loss_dict_epoch:
                    loss_dict_epoch[key] = value.detach()
                else:
                    loss_dict_epoch[key] += value.detach()

            cfg.scaler.scale(loss_total).backward()
            cfg.scaler.step(optimizer)
            cfg.scaler.update()
            
            if cfg.writer and dist.get_rank() == 0:
                for loss_name, loss_value in loss_dict.items():
                    cfg.writer.add_scalar(f'Train/{loss_name}', loss_value.item(), epoch * len(train_dataloader) + iter)

            if device_id == 0:
                t.update(1)

    loss_epoch /= (iter + 1)
    dist.all_reduce(loss_epoch)
    scheduler.step()

    for k, v in loss_dict_epoch.items():
        loss_dict_epoch[k] /= (iter + 1)
        dist.all_reduce(loss_dict_epoch[k])

    if cfg.rank == 0:
        logger.info({'Loss': loss_epoch / cfg.world_size})
        t.set_description(desc="Epoch %i"%epoch)
        t.set_postfix(steps=iter, loss=(loss_epoch / cfg.world_size).data.item())
        avg_loss_dict = {k: (v / cfg.world_size).item() for k, v in loss_dict_epoch.items()}
        logger.info({"Epoch {} Detailed Losses".format(epoch): avg_loss_dict})

def validate_one_epoch(
    cfg,
    model,
    device_id,
    dataset_name,
    val_dataloader,
    epoch,
    logger):
    
    omni3d_result = []
    model.eval()
    
    val_dataloader.sampler.set_epoch(epoch)

    with tqdm(total=len(val_dataloader)) as t:
        for iter, data in enumerate(val_dataloader):

            input_dict = {}
            input_dict['images'] = data['images']
            input_dict['images_shape'] = data['before_pad_size']
            input_dict['vit_pad_size'] = data['vit_pad_size']
            image_h, image_w = int(data['before_pad_size'][0, 0]), int(data['before_pad_size'][0, 1])
            
            if len(data['prepare_for_dsam']) > 0:
                gt_bboxes_2d = torch.stack([data['prepare_for_dsam'][i]['bbox_2d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                gt_bboxes_3d = torch.stack([data['prepare_for_dsam'][i]['bbox_3d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                gt_center_2d = torch.stack([data['prepare_for_dsam'][i]['center_2d'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                if cfg.inference_with_point_prompt:
                    input_dict['point_coords'] = torch.stack([data['prepare_for_dsam'][i]['point_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                if cfg.inference_with_box_prompt:
                    input_dict['boxes_coords'] = torch.stack([data['prepare_for_dsam'][i]['boxes_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])
                
            input_dict["image_for_dino"] = data["image_for_dino"].to(device_id)

            only_2d_mode = len(data['prepare_for_dsam']) > 0 and torch.all(gt_bboxes_3d == -1).item()

            if cfg.provide_gt_intrinsics_inference:
                input_dict['gt_intrinsic'] = data['K']

            ret_dict = model(input_dict)
            K_for_convert = ret_dict.get('gt_intrinsic', ret_dict['pred_K']).to(device_id)
            
            if len(data['prepare_for_dsam']) > 0:

                pred_masks = ret_dict['masks']
                iou_predictions = ret_dict['iou_predictions']
                
                # decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K_for_convert)
                if cfg.dataset.zero_shot_dataset:
                    decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes_virtual_to_real(ret_dict, cfg, data['K'], ret_dict['pred_K'])
                else:
                    decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K_for_convert)
                bboxes_pred_2d_center_x = decoded_bboxes_pred_2d[..., 0]
                bboxes_pred_2d_center_y = decoded_bboxes_pred_2d[..., 1]

                rot_mat = None
                gt_rot_mat = None
                if cfg.output_rotation_matrix:
                    rot_mat = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])
                    
                    gt_rot_mat = torch.stack([data['prepare_for_dsam'][i]['rotation_pose'].to(device_id) for i in range(len(data['prepare_for_dsam']))])

                if cfg.add_cubercnn_for_ap_inference:

                    pred_pose_for_cubercnn = calculate_pred_pose_for_cubercnn(decoded_bboxes_pred_3d, rot_mat, device_id)
                    decoded_bboxes_pred_3d_corners = compute_3d_bbox_vertices_batch(decoded_bboxes_pred_3d, rot_mat)
                    new_order = torch.tensor([5, 1, 0, 4, 6, 2, 3, 7]).to(device_id)

                    decoded_bboxes_pred_3d_corners_for_cubercnn = decoded_bboxes_pred_3d_corners[:, new_order, :]
                    

                    for i in range(len(data['prepare_for_dsam'])):
                        dict_i = {}

                        dict_i['image_id'] = data['prepare_for_dsam'][i]['image_id']
                        todo_box2d = data['prepare_for_dsam'][i]['bbox_2d'].numpy()
                        resize_ratio = max(data['original_size'].squeeze()) / max(image_h, image_w)
                        dict_i['bbox'] = (todo_box2d*resize_ratio.item()).tolist()
                        dict_i['bbox'][2] = dict_i['bbox'][2] - dict_i['bbox'][0]
                        dict_i['bbox'][3] = dict_i['bbox'][3] - dict_i['bbox'][1]
                        dict_i['category_id'] = data['prepare_for_dsam'][i]['label']
                        dict_i['score'] = data['prepare_for_dsam'][i]['score']
                        dict_i['depth'] = decoded_bboxes_pred_3d[i, 2].cpu().numpy().tolist()
                        dict_i['bbox3D'] = decoded_bboxes_pred_3d_corners_for_cubercnn[i].cpu().numpy().tolist()
                        dict_i['center_cam'] = decoded_bboxes_pred_3d[i, :3].cpu().numpy().tolist()
                        dict_i['center_2D'] = [bboxes_pred_2d_center_x[i].cpu().numpy().tolist(), bboxes_pred_2d_center_y[i].cpu().numpy().tolist()]
                        dict_i['pose'] = pred_pose_for_cubercnn[i].cpu().numpy().tolist()
                        dict_i['dimensions'] = decoded_bboxes_pred_3d[i, 3:6].cpu().numpy().tolist()
                        dict_i['area'] = dict_i['bbox'][2] * dict_i['bbox'][3] 
                        dict_i['yaw'] = decoded_bboxes_pred_3d[i, 6].cpu().numpy().tolist()
                        omni3d_result.append(dict_i)

                if cfg.model.original_sam:
                    instance_ids = [dataset_name + '_' + f"{data['prepare_for_dsam'][i]['instance_id']}" for i in range(len(data['prepare_for_dsam']))]
                    save_mask_images(pred_masks, iou_predictions ,image_h, image_w, gt_bboxes_2d, data['images'], instance_ids, save_root = cfg.exp_dir)
                # visualize
                if device_id == 0 and iter <= cfg.visualize_num:

                    input_dict['point_coords'] = torch.stack([data['prepare_for_dsam'][i]['point_coords'].to(device_id) for i in range(len(data['prepare_for_dsam']))])

                    vis_prompt_func(
                        cfg, data['images'], 
                        input_dict['point_coords'][:, 0, :], 
                        K_for_convert, data['K'], 
                        image_h, image_w, 
                        pred_masks, 
                        gt_bboxes_2d, 
                        decoded_bboxes_pred_2d, 
                        gt_bboxes_3d, 
                        decoded_bboxes_pred_3d, 
                        gt_center_2d,
                        ret_dict['pred_center_2d'],
                        gt_rot_mat,
                        rot_mat,
                        dataset_name,
                        epoch, iter, ret_dict.get('pred_box_ious', None))
    
            intrinsic_gt = data['K']
            intrinsic_pred = ret_dict['pred_K']

            depth_map = ret_dict['depth_maps'][:, :image_h, :image_w]
            depth_gt = data['depth'].to(device_id)[:, :image_h, :image_w]
            masks = data['masks'].to(device_id)[:, :image_h, :image_w]                
            
            if iter < 5 and device_id == 0:
                depth_gt[depth_gt == torch.inf] = 0

                if (epoch + 1) % cfg.eval_interval == 0:
                    save_depth_image(depth_gt[0].unsqueeze(0), f'{cfg.exp_dir}/{dataset_name}_depth_gt_{iter}.png', max_depth=int(depth_gt[0].max()))
                    save_color_image(data['images'][0].unsqueeze(0), image_h, image_w, f'{cfg.exp_dir}/{dataset_name}_raw_gt_{iter}.png')
                    save_point_cloud(depth_gt[0].unsqueeze(0), data['images'][0].unsqueeze(0), intrinsic_gt[0].unsqueeze(0), f"{cfg.exp_dir}/{dataset_name}_gt_{iter}.ply", image_h, image_w)

                save_depth_image(depth_map[0].unsqueeze(0), f'{cfg.exp_dir}/{dataset_name}_depth_pred_{epoch}_{iter}.png', max_depth=int(depth_map[0].max()))
                save_point_cloud(depth_map[0].unsqueeze(0), data['images'][0].unsqueeze(0), K_for_convert[0].unsqueeze(0), f"{cfg.exp_dir}/{dataset_name}_pred_{epoch}_{iter}.ply", image_h, image_w)
        
            if device_id == 0:
                t.update(1) 

        if device_id == 0:
            t.set_description(desc=f"Val Epoch {epoch} for {dataset_name}")

    return omni3d_result
        

def trainval_sam(
    cfg,
    model,
    device_id,
    start_epoch,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloaders = None,
    logger = None,
    teacher_model = None,
    linear_align = None,
    channel_align = None,
    channel_reduction = None,):
    """The SAM training loop."""

    for epoch in range(start_epoch, cfg.num_epochs):
        if not cfg.inference_only:
            train_one_epoch(
                cfg,
                model,
                device_id,
                optimizer,
                scheduler,
                train_dataloader,
                epoch,
                logger,
            )
            
            if cfg.rank == 0:
                save_checkpoint({ 
                    'epoch': epoch + 1, 
                    'state_dict': model.module.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'scheduler': scheduler.state_dict(), # 保存调度器的状态字典 
                    'scaler': cfg.scaler.state_dict(),  # 保存 AMP 状态
                    }, cfg.exp_dir, 'checkpoint_{}.pth'.format(epoch))
                    
            if (epoch + 1) % cfg.eval_interval != 0:
                continue

        with torch.no_grad():
            for dataset_name, val_dataloader in val_dataloaders:
                omni3d_result = validate_one_epoch(
                    cfg,
                    model,
                    device_id,
                    dataset_name,
                    val_dataloader,
                    epoch,
                    logger
                )
            
                if cfg.add_cubercnn_for_ap_inference:
                    dist.barrier()
                    # 每张卡都有自己的部分结果
                    local_result = omni3d_result  # 是 list 或 dict 都可以
                    all_results = [None for _ in range(dist.get_world_size())]

                    # 所有卡 gather 结果
                    all_gather_object(all_results, local_result)

                    if dist.get_rank() == 0:
                        final_results = []
                        for r in all_results:
                            final_results += r  # 或 final_results.append(r) 如果是 dict
                        with open(f"{cfg.exp_dir}/{dataset_name}_{cfg.output_json_file}.json", "w") as f:
                            json.dump(final_results, f, indent=4)
                    dist.barrier()

def main():
    os.environ['NCCL_DEBUG'] = 'INFO'
    dist.init_process_group("nccl")
    # dist.init_process_group("nccl", timeout=timedelta(seconds=7200000)) # was 1800000
    os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Start running basic DDP example on rank {rank}.")
    # device_id = rank
    args = parse_args()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = Box(cfg)

    # 使用命令行参数覆盖配置文件参数
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    time_filename = datetime.now().strftime('%m%d-%H%M%S')
    cfg.exp_dir = os.path.join(cfg.exp_dir, time_filename)
    logger = get_mylogger(log_dir=cfg.exp_dir)
    cfg.rank = rank
    cfg.world_size = world_size
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))
    else:
        writer = None
    cfg.writer = writer

    logger.info(cfg)
    transform_train, transform_test = get_depth_transform()
    
    train_dataset = DetAny3DDataset(cfg, transform = transform_train, mode = 'train')
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, collate_fn=collector)

    val_loaders = []
    for dataset_name in cfg.dataset.val.keys():
        val_dataset = DetAny3DDataset(cfg, transform=transform_test, mode='val', dataset_name=dataset_name)
        logger.info(f"val_dataset: {val_dataset.raw_info}")

        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=val_sampler, collate_fn=collector)
        val_loaders.append((dataset_name, val_loader))

    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    my_sam_model = WrapModel(cfg)
    torch.cuda.empty_cache()
    cfg.scaler = GradScaler()

    if cfg.resume:
        assert os.path.isfile(cfg.resume)
        logger.info("=> loading checkpoint '{}'".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=f'cuda:{device_id}')
        start_epoch = checkpoint['epoch']
        new_model_dict = my_sam_model.state_dict()
        for k,v in new_model_dict.items():
            if k in checkpoint['state_dict'].keys() and checkpoint['state_dict'][k].size() == new_model_dict[k].size():
                new_model_dict[k] = checkpoint['state_dict'][k].detach()
        my_sam_model.load_state_dict(new_model_dict)
        logger.info("=> loaded checkpoint '{}' (epoch {})" .format(cfg.resume, checkpoint['epoch']))

    else: 
        start_epoch = 0    
        unidepth_checkpoint = torch.load(cfg.unidepth_path, map_location=f'cuda:{device_id}')
        check_list = {k:0 for k in unidepth_checkpoint.keys()}
        new_model_dict = my_sam_model.state_dict()
        for k,v in new_model_dict.items():
            if 'sam.image_encoder.dino' in k:
                if 'pixel_encoder' + k.split('sam.image_encoder.dino')[1] in unidepth_checkpoint.keys():
                    new_key = 'pixel_encoder' + k.split('sam.image_encoder.dino')[1]
                    new_model_dict[k] = unidepth_checkpoint[new_key].detach()
                    check_list[new_key] = 1
            elif 'sam.image_encoder.depth_head' in k:
                if 'pixel_decoder' + k.split('sam.image_encoder.depth_head')[1] in unidepth_checkpoint.keys():
                    new_key = 'pixel_decoder' + k.split('sam.image_encoder.depth_head')[1]
                    new_model_dict[k] = unidepth_checkpoint[new_key].detach()
                    check_list[new_key] = 1
        my_sam_model.load_state_dict(new_model_dict)

    my_sam_model.to(device_id)
    my_sam_model.setup()

    my_sam_model = DDP(my_sam_model, device_ids=[device_id], find_unused_parameters=True)

    optimizer, scheduler = configure_opt(cfg, my_sam_model, train_loader)
    
    if cfg.resume and not cfg.inference_only:
        if cfg.resume_scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            cfg.scaler.load_state_dict(checkpoint['scaler']) # 加载 AMP 状态

    trainval_sam(cfg, my_sam_model, device_id, start_epoch, optimizer, scheduler, train_loader, val_dataloaders = val_loaders, logger = logger)


if __name__ == '__main__':  
    main()