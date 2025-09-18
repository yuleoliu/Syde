import os
import sys
import time
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from PIL import Image
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from ttda_method import ZeroShotCLIP,ZeroShotNTTA_queue, ZeroShotNTTA, TPT, Tent, SoTTA, TDA, ZeroShotNTTA_DEYO,ZeroShotNTTA_DEYO_gap

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.datautils import AugMixAugmenter
from data.build_dataset import build_test_data
from clip.classifier import *
from utils.utils import *
# from utils.losses import *
from data.dataset_class_names import get_classnames


def init_update_config():

    FLAGS.config.data.test_set = FLAGS.test_set
    FLAGS.config.data.OOD_set = FLAGS.OOD_set
    FLAGS.config.gpu = FLAGS.gpu
    FLAGS.config.training.plpd_threshold = FLAGS.plpd_threshold 
    FLAGS.config.training.deyo_margin = FLAGS.deyo_margin 

    FLAGS.config.training.size = FLAGS.size 

    FLAGS.config.training.filter_ent = FLAGS.filter_ent 
    FLAGS.config.training.filter_plpd = FLAGS.filter_plpd 
    FLAGS.config.training.gap = FLAGS.gap 
    FLAGS.config.training.patch_size = FLAGS.patch_size

    FLAGS.config.training.cache_len = FLAGS.cache_len

    FLAGS.config.training.som_lr = FLAGS.som_lr
    FLAGS.config.training.som_iter = FLAGS.som_iter
    FLAGS.config.training.vis_weight = FLAGS.vis_weight
    FLAGS.config.training.som_sigma = FLAGS.som_sigma



    # for tent setup
    if FLAGS.config.data.test_set in ['I', 'K', 'A', 'R', 'V'] and FLAGS.config.method == "Tent":
        FLAGS.config.optim.optimizer = "SGD" # SGD, Adam, AdamW
        FLAGS.config.optim.lr = 0.00025
        FLAGS.config.optim.weight_decay = 0
        FLAGS.config.optim.momentum = 0.9  
    
    # Construct the base experiment_id
    if FLAGS.config.data.OOD_ratio > 0:
        experiment_id = (
            f"{FLAGS.config.method}_{FLAGS.config.data.test_set}_"
            f"{FLAGS.config.model.arch.replace('/', '_')}_{FLAGS.config.data.OOD_set}_"
            f"{FLAGS.config.data.OOD_ratio}_{FLAGS.config.anlysis_mode}_"
            f"bs_{FLAGS.config.inference.batch_size}_"
            f"{FLAGS.config.inference.threshold_type}"
        )
    else:
        experiment_id = (
            f"{FLAGS.config.method}_{FLAGS.config.data.test_set}_"
            f"{FLAGS.config.model.arch.replace('/', '_')}_"
            f"clean_{FLAGS.config.data.OOD_ratio}_{FLAGS.config.anlysis_mode}_"
            f"bs_{FLAGS.config.inference.batch_size}_"
            f"{FLAGS.config.inference.threshold_type}"
        )
    if FLAGS.config.inference.threshold_type == "fixed":
        experiment_id += f"_{FLAGS.config.inference.fixed_threshold}"

    elif FLAGS.config.method == "ZS-NTTA":
        experiment_id += f"_ttda_step_{FLAGS.config.inference.using_ttda_step}_plpd_{FLAGS.config.training.plpd_threshold}_deyo_{FLAGS.config.training.deyo_margin}_ent_plpd_{FLAGS.config.training.filter_ent}_gap_{FLAGS.config.training.gap}_som_size_{FLAGS.config.training.size}_patch_shuffle_{FLAGS.config.training.patch_size}_cache_lens_{FLAGS.config.training.cache_len}_som_lr{FLAGS.config.training.som_lr}_som_iter{FLAGS.config.training.som_iter}_vis_weight_{FLAGS.config.training.vis_weight}_som_sigma_{FLAGS.config.training.som_sigma}"

    FLAGS.config.logs.experiment_id = experiment_id


def main(argv):
    print(FLAGS.config)
    init_update_config()
    set_random_seed(FLAGS.config.seed)

    result_path = (
        f"{FLAGS.config.logs.path}/"
        f"{FLAGS.config.logs.experiment_group}/"
        f"{FLAGS.config.data.test_set}/"
        f"{FLAGS.config.method}_other_id/"
        "mcm_filter/"
        f"{FLAGS.config.logs.experiment_id}.txt"
        )

    if os.path.exists(result_path):
        print("========================= file exists, below is the result =========================")
        with open(result_path, 'r') as f:
            print(f.read())
        print(f"========================= file exists: {result_path} =========================")

        skipped_log_path = os.path.join(FLAGS.config.logs.path, "logs", "skipped_experiments.log")
        os.makedirs(os.path.dirname(skipped_log_path), exist_ok=True)
        with open(skipped_log_path, 'a') as log_file:
            log_file.write(result_path + "\n")
    else:
        print("========================= perform TTDA pipeline =========================")
        main_worker(FLAGS.config)


def main_worker(args):
    print("Use GPU: {} for training".format(args.gpu))
    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    classnames = get_classnames(args.data.test_set)
    args.class_num = len(classnames)
    print(f'=> loaded {args.data.test_set} classname')

    if args.method == "ZS-CLIP":
        learner_method = ZeroShotCLIP
    elif args.method == "ZS-NTTA":
        learner_method = ZeroShotNTTA_DEYO_gap
    elif args.method == "TPT":
        learner_method = TPT
    elif args.method == "Tent":
        learner_method = Tent
    elif args.method == "SoTTA":
        learner_method = SoTTA
    elif args.method == "TDA":
        learner_method = TDA
    elif args.method == "ZS-NTTA-gap":
        learner_method = ZeroShotNTTA_DEYO_gap
    else:
        raise NotImplementedError
    
    learner = learner_method(args)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    results = {}
    # for set_id in datasets:
    if args.method == 'TPT':
        base_transform = transforms.Compose([
            transforms.Resize(args.model.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.model.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.inference.batch_size-1, 
                                        augmix=len(args.data.test_set)>1)
        data_corrupt_transform = AugMixAugmenter(None, preprocess, n_views=args.inference.batch_size-1, 
                                        augmix=len(args.data.test_set)>1)
        batchsize = 1
    else:
        data_transform = transforms.Compose([
            transforms.Resize(args.model.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.model.resolution),
            transforms.ToTensor(),
            normalize,
        ])
        data_corrupt_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        batchsize = args.inference.batch_size

    print("evaluating: {}".format(args.data.test_set))

    val_dataset, id_teset = build_test_data(args, data_transform, data_corrupt_transform)
    print("number of test samples: {}".format(len(val_dataset)))

    id_loader = torch.utils.data.DataLoader(
        id_teset,
        batch_size=batchsize, shuffle=False,
        num_workers=args.data.workers, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batchsize, shuffle=True,
                num_workers=args.data.workers, pin_memory=True)
    # learner.model.get_id_visual_classifier(id_loader,args)
    results[args.data.test_set] = test_time_adapt_eval(val_loader, learner, args)
    del val_dataset, val_loader
    try:
        print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(args.data.test_set, results[args.data.test_set][0], results[args.data.test_set][1]))
    except:
        print("=> Acc. on testset [{}]: {}".format(args.data.test_set, results[args.data.test_set]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("\t\t [test_set] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")
    
    result_path = f"{args.logs.path}/{args.logs.experiment_group}/{args.data.test_set}/{args.method}/"
    os.makedirs(result_path, exist_ok=True)
    with open(f'{result_path}{args.logs.experiment_id}.txt', 'w') as f:
        f.write(str(args) + '\n')
        for id in results.keys():
            if args.data.OOD_set != 'None':
                f.write("ACC_S: {:.2f}\t".format(results[id][0]))
                f.write("ACC_N: {:.2f}\t".format(results[id][1]))
                f.write("ACC_H: {:.2f}\t".format(results[id][2]))
                f.write("AUROC: {:.2f}\t".format(results[id][3]))
                f.write("AUPR: {:.2f}\t".format(results[id][4]))
                f.write("FPR95: {:.2f}\t".format(results[id][5]))
                f.write("Time: {:.2f}\t".format(results[id][6]))
            else:
                f.write("ACC: {:.2f}\t".format(results[id][0]))
        f.write("\n\n\n\n")

def print_gpu_memory():
    if torch.cuda.is_available():
        device = torch.device("cuda:3")
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # 已分配显存 (MB)
        reserved = torch.cuda.memory_reserved(device) / 1024**2    # 已保留显存 (MB)
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2  # 运行过程中的最大值
        print(f"Allocated: {allocated:.2f} MB")
    else:
        print("CUDA is not available.")

def test_time_adapt_eval(val_loader, learner, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix='Test: ')
    end = time.time()

    correct = []
    unseen_correct= []
    all_correct=[]
    num_open = 0
    num_id_err = 0
    num_ood_err = 0
    predicted_list=[]
    label_list=[]

    conf_ood = []
    conf_id = []
    class_num = args.class_num
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)
        
        learner.setup(images, target)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                st = time.time()
                output, image_feature_raw = learner.get_output(image) 
                assert output.dim() == 2
                # _, predicted = output.max(dim=-1)

                

                logit_text = F.softmax(output, dim=1)

                # entropys,plpd =  learner.get_ent_plpd(output, image, image_feature_raw, i, target)
                # if i == 0:
                #     plpd_list = plpd
                #     entropys_list = entropys
                #     target_list = target
                # else:
                #     plpd_list = torch.cat([plpd_list,plpd])
                #     entropys_list = torch.cat([entropys_list,entropys])
                #     target_list = torch.cat([target_list,target])

                
                # np.save(f'plpd_list_{args.data.OOD_set}.npy',plpd_list.cpu().numpy())
                # np.save(f'entropy_list_{args.data.OOD_set}.npy',entropys_list.cpu().numpy())
                # np.save(f'target_list_{args.data.OOD_set}.npy',target_list.cpu().numpy())

                # continue

                # if args.method == 'ZS-NTTA' and args.inference.batch_size == 1:
                #     ttda_queue_length = args.inference.ttda_queue_length
                # else:
                ttda_queue_length = args.inference.batch_size
                
                if i * args.inference.batch_size > args.inference.using_ttda_step * ttda_queue_length:
                    unseen_mask, conf, visual_output = learner.get_unseen_mask(output, image, image_feature_raw, i, target)
                else:
                    unseen_mask, visual_output,plpd, entropys, target = learner.get_unseen_mask(output, image, image_feature_raw, i, target)
                    logit = F.softmax(output, dim=1)
                    conf, _ = logit.max(dim=-1)
                print('FPS:',128/(time.time()-st))

                # if i == 0:
                #     plpd_list = plpd
                #     entropys_list = entropys
                #     target_list = target
                # else:
                #     plpd_list = torch.cat([plpd_list,plpd])
                #     entropys_list = torch.cat([entropys_list,entropys])
                #     target_list = torch.cat([target_list,target])
                # print(plpd_list.shape,entropys_list.shape,target_list.shape)

                # np.save(f'plpd_list_{args.data.OOD_set}.npy',plpd_list.cpu().numpy())
                # np.save(f'entropy_list_{args.data.OOD_set}.npy',entropys_list.cpu().numpy())
                # np.save(f'target_list_{args.data.OOD_set}.npy',target_list.cpu().numpy())

                logit_vis = F.softmax(visual_output[:,:class_num], dim=1)
                if i * args.inference.batch_size < args.inference.using_ttda_step * ttda_queue_length:
                    logit_all = logit_text
                else:
                    logit_all = (1-args.training.vis_weight) * logit_text + args.training.vis_weight * logit_vis
                conf_vis, predicted = logit_all.max(dim=-1)  # 1x10
                # if args.method == 'TDA':
                #     output = learner.run_test_tda(image_feature_raw, output)
                #     _, predicted = output.max(dim=-1)
                #
                #
                # if args.method == 'ZS-NTTA' and args.inference.update_classifier == 'TDA':
                #     output = learner.run_test_tda(image_feature_raw, output)
                #     _, predicted = output.max(dim=-1)

                valid_mask = target >= 0
                conf_ood.extend(conf[(target == class_num) & valid_mask].detach().cpu().tolist())
                conf_id.extend(conf[(target < class_num) & valid_mask].detach().cpu().tolist())

                predicted[unseen_mask & valid_mask] = class_num

                one = torch.ones_like(target) * class_num
                false = torch.ones_like(target) * -1

                seen_labels = torch.where((target > class_num - 1) & valid_mask, false, target)
                unseen_labels = torch.where((target > class_num - 1) & valid_mask, one, false)
                all_labels = torch.where((target > class_num - 1) & valid_mask, one, target)

                correct.append(predicted[valid_mask].eq(seen_labels[valid_mask]))
                unseen_correct.append(predicted[valid_mask].eq(unseen_labels[valid_mask]))
                all_correct.append(predicted[valid_mask].eq(all_labels[valid_mask]))

                num_open += torch.gt(target[valid_mask], class_num - 1).sum()
                
                id_err = ((target < class_num) & (predicted == class_num) & valid_mask).sum()
                ood_err = ((target == class_num) & (predicted < class_num) & valid_mask).sum()
                num_id_err += id_err
                num_ood_err += ood_err

                predicted_list.append(predicted.long().cpu())
                label_list.append(all_labels.long().cpu())



        if args.data.OOD_set != 'None':
            print(len(torch.cat(correct).cpu().numpy()))
            seen_acc = round(torch.cat(correct).cpu().numpy().sum() / (len(torch.cat(correct).cpu().numpy()) - num_open.cpu().numpy()), 4)
            unseen_acc = round(torch.cat(unseen_correct).cpu().numpy().sum() / num_open.cpu().numpy(), 4)
            h_score = round((2 * seen_acc * unseen_acc) / (seen_acc + unseen_acc), 4)
            print(f'Batch: ({i}/{len(val_loader)})\t Cumulative Results: '
                f'ACC_S: {seen_acc * 100:.2f}\tACC_N: {unseen_acc * 100:.2f}\t'
                f'ACC_H: {h_score * 100:.2f}\tID: {len(torch.cat(correct).cpu().numpy()) - num_open.cpu().numpy():.2f}\t'
                f'OOD: {num_open.cpu().numpy():.2f}, id_err: {num_id_err}, ood_err: {num_ood_err}')
        else:
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5)) # output: [bs, cls]
                    
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            progress.display(i)


    if args.method == "Tent":
        learner.writer.close()

    if args.data.OOD_set != 'None':
        if args.inference.threshold_type == 'adaptive':
            conf_to_save = {
                'conf_ood': conf_ood,
                'conf_id': conf_id
            }
            conf_path = f"{args.logs.conf_path}/{args.logs.experiment_group}/{args.data.test_set}/{args.method}/"
            os.makedirs(conf_path, exist_ok=True)
            pkl_save_path = f'{conf_path}/{args.logs.experiment_id}.pkl'
            save_pkl(pkl_save_path, conf_to_save)

            if args.data.OOD_ratio > 0 or args.method == 'ZS-NTTA':
                plot_distribution(args, conf_id, conf_ood, args.logs.experiment_id)

        if args.data.OOD_ratio > 0:
            auroc, aupr, fpr = get_measures(conf_id, conf_ood)
        else:
            auroc, aupr, fpr = 0, 0, 0
        run_time = time.time() - end
        return [seen_acc * 100, unseen_acc * 100, h_score * 100, auroc * 100, aupr * 100, fpr * 100, run_time]
    else:
        progress.display_summary()
        return [top1.avg, top5.avg]


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string(name='test_set', default='CIFAR-10', help='ID dataset.')
flags.DEFINE_string(name='OOD_set', default='SVHN', help='OOD dataset.')
flags.DEFINE_integer(name='gpu', default=0, help='gpu number.')
flags.DEFINE_float(name='plpd_threshold', default=1e-4, help='plpd_threshold.')
flags.DEFINE_float(name='deyo_margin', default=2.0, help='deyo_margin.')
flags.DEFINE_integer(name='size', default=15, help='som_size.')
flags.DEFINE_integer(name='filter_ent', default=1, help='ent.')
flags.DEFINE_integer(name='filter_plpd', default=1, help='plpd.')
flags.DEFINE_integer(name='gap', default=1, help='gap.')
flags.DEFINE_integer(name='patch_size', default=20, help='patch_size.')
flags.DEFINE_integer(name='cache_len', default=10, help='patch_size.')
flags.DEFINE_float(name='som_lr', default=0.05, help='som_lr.')
flags.DEFINE_integer(name='som_iter', default=100, help='som_iter.')
flags.DEFINE_float(name='vis_weight', default=0.1, help='vis_weight.')
flags.DEFINE_integer(name='som_sigma', default=2, help='som_sigma.')


if __name__ == '__main__':
    app.run(main)