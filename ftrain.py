from datasets import *
from models import *
from Learning.learning import *
from Learning.losses import *
from Utils.parser_ import get_args
from fastai2.vision.all import *
from Learning.learning import test
from utils_ import measure_time
import datetime
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import csv
import gc

@dataclass
class Do_everything(Callback):
    learn: Learner

    def begin_epoch(self):  # on_epoch_begin
        L = self.learn
        if hasattr(self, 'skip_prepare') and self.skip_prepare:
            return
        L.dls.loaders[0].prepare_epoch()

    def begin_batch(self):  # before the output and loss are computed. # on_batch_begin
        L = self.learn
        self.info = L.yb

    def after_pred(self):  # after forward pass but before loss has been computed # on_loss_begin
        pass

    def after_loss(self): # after the forward pass and the loss has been computed, but before backprop # on_backward_begin
        L = self.learn
        if L.epoch != L.pre_epoch and (args.loss=='tripletMargin++' or args.loss=='tripletMargin_o' or args.loss=='ExpTeacher'):
            a = L.loss[1][0]; b = L.loss[1][1]
            B = L.loss[1][2]
            if args.loss=='ExpTeacher' or args.loss=='tripletMargin++' :
                confident_weight = L.loss[1][4]
                writer.add_histogram(tag='confident_weight/' + args.id, values=confident_weight, global_step=L.pre_epoch - 1)
            writer.add_scalar(tag='B', scalar_value=B, global_step=L.pre_epoch - 1)
            # writer.add_scalar(tag='C', scalar_value=C, global_step=L.pre_epoch - 1)
            writer.add_histogram(tag='a_data/'+args.id, values=a, global_step=L.pre_epoch - 1)
            writer.add_histogram(tag='b_data/'+args.id, values=b, global_step=L.pre_epoch - 1)

            L.pre_epoch = L.epoch
        L.loss = L.loss[0]

    def after_train(self):  # on_epoch_end
        L = self.learn
        if (args.all_info or L.epoch==args.epochs-1):
            model_dir = pjoin(args.model_dir, save_name)
            os.makedirs(model_dir, exist_ok=True)
            save_path = pjoin(model_dir, 'checkpoint_{}.pt'.format(L.epoch))
            printc.green('saving to: {} ...'.format(save_path))
            if L.model.name == 'TeacherNet' and L.epoch != 0:
                if L.model.teacher == 'self':
                    torch.save({ 'S_state_dict': L.model.student.state_dict(),
                                'T_state_dict': L.model.student.state_dict(), 'model_arch': L.model.name,
                                 'S_model_arch': L.model.student.name,
                                 'T_model_arch': L.model.teacher.name,
                                 'save_name': 'Self-TNet'}, save_path)
                else:
                    torch.save({'epoch': L.epoch + 1, 'T_state_dict': L.model.teacher.state_dict(),
                                'S_state_dict': L.model.student.state_dict(),
                                'model_arch': L.model.name, 'S_model_arch': L.model.student.name,
                                'T_model_arch': L.model.teacher.name,
                                'save_name': save_name}, save_path)
            else:
                torch.save({'epoch': L.epoch + 1, 'state_dict': L.model.state_dict(), 'model_arch': L.model.name, 'save_name':save_name}, save_path)
            dst = pjoin(model_dir, '{}.pt'.format(save_name))

            L.writer.add_scalars('FPR95_sum', {'sum_FPR95': L.FPR_sum}, L.epoch)
            L.FPR_sum = -1

            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.relpath(save_path, getdir(dst)), dst)

    def begin_validate(self):  # on_epoch_end
        L = self.learn
        if not args.notest:
            for test_loader in L.test_loaders:
                test(test_loader, L.model, test_loader.dataset.name,  L.epoch, L.writer, L)

        raise CancelValidException()


def loss_generalized(name:str, output, info):
    labels = info['labels'].long().cuda()
    if isinstance(output, type(torch.zeros(0))):
        if 'h8' in info["model_arch"] or 'h7' in info["model_arch"] or 'SDGMNet' in info["model_arch"]:
            cross_loss, edge = tripletMargin_generalized_Exponential(
                embeddings=output, labels=labels, neg_num=args.neg_num, margin_pos=args.marginpos,
                R=args.R, D=args.D, B=args.B, A=args.A, use_stB=args.use_stB,
                threshold=args.threshold, is_finetune=args.use_finetune, ranges=args.range
            )
            return cross_loss,  edge
        else:
            raise "output ERROR"
    else:
        raise "output ERROR"



def tripletMargin_o_loss(name:str, output, info):
    labels = info['labels'].long().cuda()
    loss, edge = tripletMargin_generalized(embeddings=output, labels=labels, margin_pos=args.marginpos)
    return loss, edge

def ExpTeacher_loss(name:str, output, info):
    labels = info['labels'].long().cuda()
    # assert info["model_arch"] == 'TeacherNet'
    sum_loss = 0
    N = len(labels)
    used_neg_dist = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
    for i in range(args.neg_num):
        cross_loss_final, edge = tripletMargin_generalized_ExpTeacher(
            used_neg_dist=used_neg_dist, embeddings=output['student'],
            embeddingsT=output['teacher'],
            labels=labels, R=args.R,
            B=args.B, A=args.A, use_stB=args.use_stB,
            threshold=args.threshold,
            upper=args.upper
        )
        sum_loss = sum_loss + cross_loss_final
    return sum_loss/args.neg_num, edge


def load_model(model_arch):
    if args.resume != '':
        printc.green('studentNet Arch:')
        model = get_model_by_name(model_arch).cuda()
        printc.green('Loading resume model:', args.resume, '...')
        saved_model = load_hardnet(args.resume, strict=True)
        if saved_model.name == 'TeacherNet':
            model.load_state_dict(saved_model.student.state_dict(), strict=True)
        else:
            model.load_state_dict(saved_model.state_dict(), strict=True)
        if hasattr(saved_model, 'pca'):
            model.pca = saved_model.pca
    else:
        printc.green('studentNet Arch:')
        model = get_model_by_name(model_arch).cuda()

    if args.teacher != '':
        assert args.loss == 'ExpTeacher'
        printc.green('Loading TeacherNet model:', args.teacher, '...')
        if args.teacher == 'self':
            teacherNet = 'self'
        else:
            teacherNet = load_hardnet(args.teacher, strict=True).cuda()
            if hasattr(teacherNet, 'student'):
                teacherNet = teacherNet.student
        studentNet = model

        model = get_model_by_name('TeacherNet').cuda()
        model.teacher = teacherNet
        model.student = studentNet
        if model.teacher != 'self':
            for name, param in model.teacher.named_parameters():
                if "XXX" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    return model


def get_loss_function(name, num_classes=None, embedding_size=None, miner=None):
    if name in 'tripletMargin_o':
        printc.red('using tripletMargin_o')
        loss = partial(tripletMargin_o_loss,  name)
    elif name in 'ExpTeacher':
        printc.red('using ExpTeacher_loss')
        loss = partial(ExpTeacher_loss, name)
    elif name[-2:]=='++':
        printc.red('using generalized loss')
        loss = partial(loss_generalized, name[:-2])
    else:
        raise ValueError('Unknow loss')

    return loss

def run_AT(model_arch, test_loaders):
    model = load_model(model_arch)
    loss = get_loss_function(args.loss, miner=args.miner, embedding_size=model.osize if hasattr(model,'osize') else None)

    anneal_cycle = (args.batch_size - 1024) // 128  # if bs=2560 size=256 then cycle=7, end bs=768
    lr_list = [args.lr * args.lr_factor ** i for i in range(anneal_cycle)]
    bs_list = [args.batch_size - 128 * (i + 1) for i in range(anneal_cycle)]
    tps_list = [args.bsNum * bs_list[i] for i in range(anneal_cycle)]
    thr_list = [args.threshold + 0.05 * i for i in range(anneal_cycle)]
    for i in (range(anneal_cycle)):
        args.batch_size = bs_list[i]
        args.tuples = tps_list[i]
        args.threshold = thr_list[i]
        args.seed = random.randint(1, 100)
        train_loader = get_train_dataset(args, data_name).init(args.model_dir, save_name, args=args)  # ?
        data = DataLoaders(train_loader, test_loaders[0])
        if args.optimizer == 'sgd':
            L = Learner(data, model, loss_func=loss, metrics=[], cbs=[Do_everything], opt_func=SGD, wd=0.05,
                        wd_bn_bias=False)  # opt_func=SGD,
        elif args.optimizer == 'adam':
            L = Learner(data, model, loss_func=loss, metrics=[], cbs=[Do_everything], wd=args.wd,
                        wd_bn_bias=False)  # opt_func=Adam,
        L.test_loaders = test_loaders
        L.writer = writer
        L.FPR_sum = -1
        L.pre_epoch = i + 1

        L.fit_flat_cos(1, lr_list[i], pct_start=0.6)
        del train_loader, L, data
        gc.collect()


def main(train_loader, test_loaders, model_arch):
    model = load_model(model_arch)
    data = DataLoaders(train_loader, test_loaders[0])
    loss = get_loss_function(args.loss, train_loader.total_num_labels, 
                             miner=args.miner, 
                             embedding_size=model.osize if hasattr(model,'osize') else None)

    if args.optimizer=='sgd':
        L = Learner(data, model, loss_func=loss, metrics=[], cbs=[Do_everything], opt_func=SGD, wd=0.05, wd_bn_bias=False) # opt_func=SGD,
    elif args.optimizer=='adam':
        L = Learner(data, model, loss_func=loss, metrics=[], cbs=[Do_everything], wd=args.wd, wd_bn_bias=False) #opt_func=Adam,
    L.test_loaders = test_loaders
    L.writer = writer
    L.pre_epoch = -1
    L.FPR_sum = -1

    if args.use_finetune:
        anneal_cycle = (args.batch_size - 1024) // 128  # if bs=2560 size=256 then cycle=7, end bs=768
        lr_list = [args.lr * args.lr_factor ** i for i in range(anneal_cycle)]
        bs_list = [args.batch_size - 128 * (i + 1) for i in range(anneal_cycle)]
        tps_list = [args.bsNum * bs_list[i] for i in range(anneal_cycle)]
        thr_list = [args.threshold + 0.05*i  for i in range(anneal_cycle)]
        for i in (range(anneal_cycle)):
            args.batch_size = bs_list[i]
            args.tuples = tps_list[i]
            args.threshold = thr_list[i]
            args.seed = random.randint(1,100)
            train_loader = get_train_dataset(args, data_name).init(args.model_dir, save_name, args=args)    #?
            data = DataLoaders(train_loader, test_loaders[0])
            L.dls = data
            L.pre_epoch = i + 1

            L.fit_flat_cos(1, lr_list[i], pct_start=0.6)
    else:
        L.fit_one_cycle(args.epochs, args.lr, div=25)

if __name__ == '__main__':
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%m%d-%H_%M_%S")
    with measure_time():
        args, data_name, save_name = get_args()
        writer = SummaryWriter("test_record/id:{}_Arch:{}_Time:{}".format( args.id, args.model_arch, current_time))
        become_deterministic(args.seed)
        print('data_name:', data_name, '\nsave_name:', save_name, '\n')

        # test_loaders = get_test_loaders(['liberty', 'notredame'], args.test_batch_size, args.model_arch)
        test_loaders = get_test_loaders(['liberty'], args.test_batch_size, args.model_arch)
        # test_loaders = get_test_loaders(['yosemite','notredame'], args.test_batch_size, args.model_arch)
        if args.use_finetune:
            run_AT(model_arch = args.model_arch, test_loaders=test_loaders)
        else:
            main(train_loader=get_train_dataset(args, data_name).init(args.model_dir, save_name, args=args),
                 test_loaders=test_loaders,
                 model_arch=args.model_arch)
        printc.green('--------------- Training finished ---------------')
        print('model_name:', save_name)
        writer.close()

