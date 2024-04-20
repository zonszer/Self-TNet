import argparse
from utils_ import *
import datetime

parser = argparse.ArgumentParser(description='PyTorch HardNet')
# str
parser.add_argument('--model_dir', default='Models/', help='folder to output model checkpoints')
parser.add_argument('--name', default='', help='add any name')
parser.add_argument('--id', default='7.18_MetricsNet', help='experiment id - you do not want to search the whole name')
parser.add_argument('--resume', default='', metavar='Name/path', help='path to latest checkpoint (default: none)')  #LSUV_init_bs_seed211
parser.add_argument('--teacher', default='', metavar='Name/path', help='path to teacherNet (default: none)')  #LSUV_init_bs_seed211
parser.add_argument('--optimizer', default='sgd', help='The optimizer to use (default: SGD)')
parser.add_argument('--depths', default='', help='path')
parser.add_argument('--ds', default='NIR_RGB_ref_testset', help='choose prepared mix of datasets: v3, v4, v5, ...')  #'v4+lib' NIR_RGB_ref_scale64
parser.add_argument('--init_good', default='', help='')
parser.add_argument('--model_arch', '--arch', default='SDGMNet is the alias name of HyNet', help='')  #h8, decoder, encoder
parser.add_argument('--random_tuple', default=True, help='when use 2SAE should make it False')
# parser.add_argument('--patch_gen', '--pgen', default='meanImg', help='options: oneRes, oneImg, sumImg, meanImg, medianImg, new')
parser.add_argument('--patch_gen', '--pgen', default='new', help='options: oneRes, oneImg, sumImg, meanImg, medianImg, new')
# parser.add_argument('--sigmas_v', type=str, default='v1', help='sigmas version in new detection')
parser.add_argument('--sigmas_v', default='e011', help='sigmas version in new detection')
# parser.add_argument('--loss', default='tripletMargin', help='Other options: softmax, contrastive, tripletMarginHuberInternal, face')
parser.add_argument('--loss', default='tripletMargin++', help='Other options: tripletMargin')
parser.add_argument('--miner', default='BatchHardMiner', help='')
parser.add_argument('--batch_reduce', default='min', help='Other options: average, random, random_global, L2Net')
parser.add_argument('--masks_dir', '--masks', default=None , help='')       #'Datasets/AMOS-views/AMOS-masks'
parser.add_argument('--weight_function', '--wf', default='Hessian',
                    help='Keypoints are generated with probability ~ weight function. Variants: uniform, Hessian, HessianSqrt, HessianSqrt4')
parser.add_argument('--combine', default='', help='inbatch, epoch')
# int
parser.add_argument('--epochs', '--ep', type=int, default=40, help='number of epochs to train')
parser.add_argument('--neg_num', type=int, default=1, help='number of hard neg in loss')
parser.add_argument('--tuples', '--tps', '--n_triplets', type=int, default=60000, metavar='N', help='how many tuples will generate from the dataset')
                                                                            #500000 5000000 epochtuples（5*10**6 / batch_size = batch_num per epoch)
parser.add_argument('--patch_sets', '--psets', '--tracks', type=int, default=30000, help='How many patch sets to generate. Works approximately.')
parser.add_argument('--bsNum', type=int, default=1400, help='how many batch will ues(only work in FineTune)')
parser.add_argument('--batch_size', '--bs', type=int, default=3072, metavar='BS', help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=2048, metavar='BST', help='input batch size for testing (default: 1024)')
parser.add_argument('--cams_in_batch', '--camsb', type=int, default=5, help='how many cams are source ones for a batch in AMOS')
parser.add_argument('--min_sets_per_img', '--minsets', type=int, default=-1, help='')
parser.add_argument('--max_sets_per_img', '--maxsets', type=int, default=1501, help='used in new patch_gen')
parser.add_argument('--seed', type=int, default=211, help='random seed (default: 0)')
parser.add_argument('--shear', type=int, default=25, help='augmentation: shear')
parser.add_argument('--degrees', type=int, default=25, help='augmentation: degrees')
parser.add_argument('--Npos', type=int, default=2, help='')     ##positive
# float
parser.add_argument('--lr', type=float, default=0.05, help='learning rate') #FOR SAE may be less
parser.add_argument('--marginpos', '--mpos', type=float, default=1.0, help='the margin value for the triplet loss function')
parser.add_argument('--marginneg', '--mneg', type=float, default=1.0, help='the margin value for the triplet loss function')
parser.add_argument('--R', type=float, default=1.5, help='the margin value for the triplet loss function')
parser.add_argument('--B' , type=float, default=4.0, help='the margin value for the triplet loss function')
parser.add_argument('--D' , type=float, default=0.83, help='the margin value for the triplet loss function')
parser.add_argument('--A' , type=float, default=1.0, help='the margin value for the triplet loss function')
parser.add_argument('--range' , type=float, default=0.25, help='the margin value for the triplet loss function')
parser.add_argument('--lr_factor' , type=float, default=0.8, help='the margin value for the triplet loss function')
parser.add_argument('--threshold' , type=float, default=-0.15, help='the margin value for the triplet loss function')
parser.add_argument('--upper' , type=float, default=-0.1, help='the margin value for the triplet loss function')
parser.add_argument('--good_pr', type=float, default=0.9, help='')
parser.add_argument('--gauss_s', type=float, default=30.0, help='')
parser.add_argument('--wd', type=float, default=0.1, help='weight decay (default: 0.1)')          #
parser.add_argument('--init_thr', '--thr', type=float, default=0.00016, help='thr for new sampling')
# bool
parser.add_argument('--no_fliprot', default=False, action='store_true', help='turns off flip and 90deg rotation augmentation')
parser.add_argument('--addAP', default=False, action='store_true', help='add AP lsos to standard loss')
parser.add_argument('--AP_loss', default=False, action='store_true')
parser.add_argument('--sigmas', default=False, action='store_true', help='enable sigmas')
parser.add_argument('--use_stB', default=False, action='store_true', help='enable --use_stB')
parser.add_argument('--use_finetune', default=False, action='store_true', help='enable --use_finetune')
parser.add_argument('--use_neg_weight', default=False, help='enable use_neg_weight')
# parser.add_argument('--no_masks', type=bool, default=True, action='store_true', help='')
parser.add_argument('--no_detach', default=False, action='store_true', help='detach negative desctiptors from loss grad')
parser.add_argument('--separ_batches', default=False, action='store_true', help='separates good and bad patches; sets cams_in_batch to 0')
parser.add_argument('--pairs_imgs', default=False, action='store_true', help='take positives from pairs of imgs only')
# parser.add_argument('--no_all_info', default=False, action='store_true', help='')
parser.add_argument('--all_info', default=False, action='store_true', help='')
# parser.add_argument('--new_batches', '--NB', default=False, action='store_true', help='')
parser.add_argument('--old_batches', '--NB', default=False, action='store_true', help='')
parser.add_argument('--use_patchmask', '--patchmask', default=False, action='store_true', help='do not use patch_mask')

parser.add_argument('--use_collisions', '--collisions', '--colls', default=True, help='do not use patch_mask')
parser.add_argument('--nonmax', default=False, action='store_true', help='')
parser.add_argument('--scaleit', default=False, action='store_true', help='')
parser.add_argument('--thrit', default=False, action='store_true', help='')
parser.add_argument('--to_gauss', default=False, action='store_true', help='hess peak detector back to probab map')
parser.add_argument('--AMOS_RGB', '--RGB', default=False, action='store_true', help='')
parser.add_argument('--fixed_MP', default=False, action='store_true', help='')
parser.add_argument('--only_D', default=False, action='store_true', help='')
# other / temporary
parser.add_argument('--face_margin', type=float, default=0.3, help='')
parser.add_argument('--face_scale', type=int, default=64, help='')
parser.add_argument('--gpu', type=int, default=0, help='')
parser.add_argument('--local_rank', type=int, default=0, help='')
parser.add_argument('--duplicates', type=int, default=0, help='')
parser.add_argument('--notest', default=False, action='store_true', help='')
parser.add_argument('--closeok', default=False, action='store_true', help='')
parser.add_argument('--antiaug', default=False, action='store_true', help='')
parser.add_argument('--filter_sets', default='', help='path to info file')
parser.add_argument('--aug', default='', help='augmentation')
parser.add_argument('--K', default='', type=str, help='kornia; no transforms in datasets')
parser.add_argument('--fewcams', default=False, action='store_true', help='')
parser.add_argument('--fewcams_dups', default=False, action='store_true', help='')
# parser.add_argument('--adapt', default=False, action='store_true', help='')


def get_args(ipynb=False):
    if ipynb: # for jupyter so that default args are passed
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    printc.yellow('Parsed options:\n{}\n'.format(vars(args)))

    # args.masks = not args.no_masks
    args.detach = not args.no_detach
    args.fliprot = not args.no_fliprot
    args.AMOS_GRAY = not args.AMOS_RGB

    args.new_batches = not args.old_batches
    if args.masks_dir == '':
        args.masks_dir = None

    txt = []
    if not args.patch_gen in ['new', 'sift', 'hessbaum']: txt += ['PS:' + str(args.patch_sets)]
    if args.nonmax: txt += ['nonmax']
    if args.patch_gen == 'new': txt += ['maxsets:' + str(args.max_sets_per_img)]
    # if args.patch_gen == 'new': txt += ['sigmas-v:' + args.sigmas_v]
    if args.patch_gen == 'new': txt += ['thr:' + str(args.init_thr)]
    if args.to_gauss: txt += ['gauss:' + str(args.gauss_s)]
    if args.use_stB: txt += ['use_stB:'+ str(args.use_stB)]

    if args.scaleit: txt += ['scaleit']
    if args.thrit: txt += ['thrit']

    if args.fixed_MP: txt += ['fixed-MP']

    if args.min_sets_per_img > 0: txt += ['minsets:' + str(args.min_sets_per_img)]
    txt += ['masks:' + ('None' if args.masks_dir is None else os.path.basename(args.masks_dir))]
    data_name = '_'.join(txt)

    txt = []
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%m%d-%H_%M")
    txt += ['id:' + str(args.id)]
    txt += ['T:' + current_time]
    if args.name != '': txt += ['name:' + str(args.name)]
    txt += ['arch:' + str(args.model_arch)]
    txt += ['ds:' + str(args.ds)]
    # txt += ['neg_num:' + str(args.neg_num)]
    txt += ['loss:' + args.loss.replace('_', '')]

    txt += ['R:' + str(args.R)]
    txt += ['B:' + str(args.B)]
    txt += ['lr:' + str(args.lr)]

    txt += [data_name]

    if args.resume != '': txt += ['resume']
    # if args.fliprot != '': txt += ['fliprot']
    if args.fewcams != '': txt += ['fewcams']
    txt += ['ep:' + str(args.epochs)]
    txt += ['bs:' + str(args.batch_size)]
    # txt += ['pos:' + str(args.Npos)]
    if args.combine: txt += ['comb:' + args.combine]
    # txt += ['dups:' + str(args.duplicates)]
    model_name = '_'.join([str(c) for c in txt])

    if model_name in [getbase(c) for c in glob(pjoin(args.model_dir, '*'))]:
        printc.red('WARNING: MODEL',model_name,'\nALREADY EXISTS')

    return args, data_name, model_name