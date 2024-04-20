# import itertools
import torch
from scipy import spatial
from datasets import *
# import matplotlib.pyplot as plt
from utils_ import *
from models import *
import numpy_indexed as npi
import datetime
HUGE_CONST = 1e5

become_deterministic(0)
transformation = trans_crop_resize
# transformation = trans_crop_resize64
# transformation = transform_AMOS_resize48
# transformation = transform_AMOS_resize64

def tpfp(scores, labels, numpos=None): # code from HPatches
    # count labels
    p = int(np.sum(labels))
    n = len(labels) - p

    if numpos is not None:
        assert (numpos >= p), \
            'numpos smaller that number of positives in labels'
        extra_pos = numpos - p
        p = numpos
        scores = np.hstack((scores, np.repeat(-np.inf, extra_pos)))
        labels = np.hstack((labels, np.repeat(1, extra_pos)))

    perm = np.argsort(-scores, kind='mergesort', axis=0)

    scores = scores[perm]
    # assume that data with -INF score is never retrieved
    stop = np.max(np.where(scores > -np.inf))

    perm = perm[0:stop + 1]

    labels = labels[perm]
    # accumulate true positives and false positives by scores
    tp = np.hstack((0, np.cumsum(labels == 1)))
    fp = np.hstack((0, np.cumsum(labels == 0)))
    return tp, fp, p, n, perm

def get_pr(scores, labels, numpos=None): # code from HPatches
    [tp, fp, p, n, perm] = tpfp(scores, labels, numpos)
    # compute precision and recall
    small = 1e-10
    recall = tp / float(np.maximum(p, small))
    precision = np.maximum(tp, small) / np.maximum(tp + fp, small)
    return precision, recall, np.trapz(precision, recall)

def find_train_pair(query, target, q_centers=None, t_centers=None, collisions=None, second_nearest=False):
    unique_values, idx = np.unique(t_centers, axis=0, return_index=True)
    t_centers = t_centers[idx];
    q_centers = q_centers[idx]
    query = query[idx];
    target = target[idx]
    with torch.no_grad():
        # unique_values, count = np.unique(t_centers, axis=0, return_counts=True)
        dists = torch.cdist(query, target)
        dists_centers = spatial.distance.cdist(q_centers, t_centers, 'euclidean')
        aux = np.min(dists_centers, axis=1)
        close = aux < 2.0
        # print('kept', np.sum(aux<2.0), 'removed', np.sum(aux>=2.0))
        dists = dists[close]
        dists_centers = dists_centers[close]

        idxs = dists.min(dim=1)[1].cpu().numpy()
        gt = np.argmin(dists_centers, axis=1)
        right = idxs==gt
        wrong = np.where(idxs != gt)
        pred_idx = idxs[wrong]
        real_idx = wrong; assert (gt[wrong]==wrong).all()   #because real_idx=gt[wrong]
        m_d = dists[np.arange(len(query)), idxs][wrong]
    # m_d = dists[np.arange(len(query)), idxs]  # m_d is the closest dist of each patch embeddings(133 patches ,)
    # pr, rc, ap = get_pr(-m_d, right, numpos=len(query))  # ap is obtained by integrating
    # pos_dist = torch.norm(query[wrong] - target[gt[wrong]], dim=1)   #in dim1(） compute L2 norm value
    # neg_dist = torch.norm(query[wrong] - target[idxs[wrong]], dim=1)
    return idxs, real_idx, right

def test_pair_(query, target, q_centers=None, t_centers=None, collisions=None, second_nearest=False):
    unique_values, idx = np.unique(t_centers, axis=0, return_index=True)
    t_centers = t_centers[idx];
    q_centers = q_centers[idx]
    query = query[idx];
    target = target[idx]

    dists = spatial.distance.cdist(query, target,'euclidean')
    dists_centers = spatial.distance.cdist(q_centers, t_centers,'euclidean')
    aux = np.min(dists_centers, axis=1)
    close = aux<2.0
    # print('kept', np.sum(aux<2.0), 'removed', np.sum(aux>=2.0))
    dists = dists[close]
    dists_centers = dists_centers[close]

    idxs = np.argmin(dists, axis=1)
    gt = np.argmin(dists_centers, axis=1)
    right = idxs==gt

    if second_nearest:
        dists1 = np.min(dists, axis=1)
        for i in range(len(query)): # remove the absolutely nearest ones
            dists[i,np.argmin(dists[i,:])] = sys.maxsize
        for i,a in enumerate(collisions): # remove the inconsistent ones
            dists[i, a] = sys.maxsize
        dists2 = np.min(dists, axis=1)
        # if np.sum(dists2==0)>0:
        #     print('second with dist=0 found (!)')
            # aux = np.argmin(dists[np.argmin(dists2),:])
            # print(aux)
            # print(np.argmin(dists2))

            # print(dists[np.argmin(dists2),:])
            # print(dists[np.argmin(dists2),aux-1])
            # print(dists[np.argmin(dists2),aux])
            # print(dists[np.argmin(dists2),aux+1])
            # input()
        dists2[dists2==0] = 0.000001 # but this is hack, investigate WHY DISTANCE IS ZERO
        right = right * ((dists1 / dists2) < 0.8)

    m_d = dists[np.arange(len(query)), idxs]    #m_d is the closest dist of each patch embeddings(133 patches ,)
    pr,rc,ap = get_pr(-m_d,right,numpos=len(query)) #ap is obtained by integrating
    return right, close, ap

def test_pair_changed(query, target, dists, true_label, collisions=None, second_nearest=False):
    '''used in stardard mode(images can have distortion) '''
    unique_values, idx = np.unique(t_centers, axis=0, return_index=True)
    t_centers = t_centers[idx]
    q_centers = q_centers[idx]
    query = query[idx]
    target = target[idx]

    idxs = np.argmin(dists, axis=1)
    # gt = np.argmin(dists_centers, axis=1)

    gt = np.arange(0, idxs.shape[0])
    right = idxs==gt

    if second_nearest:
        dists1 = np.min(dists, axis=1)
        for i in range(len(query)): # remove the absolutely nearest ones
            dists[i,np.argmin(dists[i,:])] = sys.maxsize
        for i,a in enumerate(collisions): # remove the inconsistent ones
            dists[i, a] = sys.maxsize
        dists2 = np.min(dists, axis=1)

        dists2[dists2==0] = 0.000001 # but this is hack, investigate WHY DISTANCE IS ZERO
        right = right * ((dists1 / dists2) < 0.8)

    m_d = dists[np.arange(len(query)), idxs]    #m_d is the closest dist of each patch embeddings(133 patches ,)
    pr,rc,ap = get_pr(-m_d,right,numpos=len(query)) #ap is obtained by integrating
    return np.where(right == True)[0].shape[0], ap

def test_pair_Anchor(query, target, q_centers=None, t_centers=None, collisions=None, second_nearest=False):
    dists = spatial.distance.cdist(query, target,'euclidean')
    dists_centers = spatial.distance.cdist(q_centers, t_centers,'euclidean')

    def get_first_criterionDist(dists, dists_centers):
        bestReg_idx = np.where(dists == np.min(dists))
        q_coord_ofbestReg = q_centers[bestReg_idx[0]]
        t_coord_ofbestReg = t_centers[bestReg_idx[1]]
        dists[bestReg_idx] = HUGE_CONST    # best_reg_dist =HUGE_CONST
        # better_reg_dist = dists[np.where(dists == np.min(dists))]
        betterReg_idx = np.where(dists == np.min(dists))
        q_coord_ofbetterReg = q_centers[betterReg_idx[0]]
        t_coord_ofbetterReg = t_centers[betterReg_idx[1]]
        q_reg_dist = sum(q_coord_ofbestReg - q_coord_ofbetterReg) ** 2
        t_reg_dist = sum(t_coord_ofbestReg - t_coord_ofbetterReg)**2
        if q_reg_dist==0 or t_reg_dist==0:
            d = get_first_criterionDist(dists, dists_centers)
            return d + 0.5
        else:
            dists_ofcriterion = q_reg_dist / t_reg_dist
            return dists_ofcriterion + 0.5

    def iteration(dists, dists_centers, first_criterionDist=1.05):
        for _ in range(dists_centers.shape[0]):
            matrix_min_idx = np.where(dists == np.min(dists))
            targetdist_withmin = dists_centers[matrix_min_idx[0]] #(get row in matrix)all target points dist with the best reg points
            posit_list= []; posit_idxlist = []
            targetdist_withmin[0, matrix_min_idx[0]] = HUGE_CONST
            dists[matrix_min_idx] = HUGE_CONST
            for i in range(dists_centers.shape[0]):
                idx = np.argmin(targetdist_withmin, axis=1)
                if i < 3 and targetdist_withmin[0, idx] > np.median(dists_centers)/2:       #@
                    break
                posit_list.append(targetdist_withmin[0, idx])
                posit_idxlist.append(idx)
                targetdist_withmin[0, idx] = HUGE_CONST  #posit_idxlist is strands for target idx
                if i >= dists_centers.shape[0]/10 and (posit_list[-2]!=0 and posit_list[-3]!=0) and posit_list[-1]/posit_list[-2] > first_criterionDist * posit_list[-2]/posit_list[-3]:    #posit_list[-4]/posit_list[-3] > 1
                # if i >= dists_centers.shape[0]/10 and (posit_list[-2]!=0 and posit_list[-4]!=0) and posit_list[-1]/posit_list[-2] > first_criterionDist * posit_list[-3]/posit_list[-4]:    #posit_list[-4]/posit_list[-3] > 1
                    # dists[matrix_min_idx[0], posit_idxlist] = HUGE_CONST
                    return matrix_min_idx, posit_idxlist[:-1]
        raise 'need change hyperparms'

    def reg_iternum(dists_changed, dists_centers, rights, APs, q_centers, t_centers, use_standardMode=False):
        if (dists_changed==HUGE_CONST).all()==True:     #@
        # if np.where(dists_changed == HUGE_CONST)[0].shape[0] > dists_changed.shape[0]*dists_changed.shape[1]//2:
            return
        else:
            if use_standardMode:
                first_criterionDist = get_first_criterionDist(dists_changed, dists_centers)
            else:
                first_criterionDist = 1.05  # is use default first_criterionDist in iteration
            bestReg_positIdx, t_posit_idxlist = iteration(dists_changed.copy(), dists_centers)  #each dists is a new one
            _, q_posit_idxlist = iteration(np.transpose(dists_changed.copy(), (1, 0)), np.transpose(dists_centers, (1, 0)), first_criterionDist)
            dists_changed[:, t_posit_idxlist] = HUGE_CONST
            dists_changed[q_posit_idxlist, :] = HUGE_CONST
            # dists_changed = np.transpose(dists_changed, (1, 0)); dists_centers = np.transpose(dists_centers,(1, 0))
            # dists_changed[bestReg_positIdx] = HUGE_CONST
            q_new = query[np.array(q_posit_idxlist).reshape(-1)]; t_new = target[np.array(t_posit_idxlist).reshape(-1)]
            # q_centers_new = q_centers[np.array(q_posit_idxlist).reshape(-1)]; t_centers_new = t_centers[np.array(t_posit_idxlist).reshape(-1)]
             # =dists_centers[np.array(q_posit_idxlist).reshape(-1), np.array(t_posit_idxlist).reshape(-1)]
            dists_NEW = spatial.distance.cdist(q_new, t_new,'euclidean')

            # true_label = np.arange(0, idxs.shape[0])
            true_label = np.where(q_posit_idxlist==t_posit_idxlist)
            right, ap = test_pair_changed(q_new, t_new, dists_NEW, true_label)
            Rights.append(right); APs.append(ap)
            return reg_iternum(dists_changed, dists_centers, rights, APs, q_centers, t_centers)
    # import sys
    # sys.setrecursionlimit(500000)
    Rights, APs = [], []
    reg_iternum(dists, dists_centers, Rights, APs, q_centers, t_centers, use_standardMode=False, )
    # test_pair_changed(query, t_new, dists, dists_centers)
    return sum(Rights)/query.shape[0], sum(APs)/len(APs)

def loss_calculate(neg_dist, pos_dist):
    loss = ((pos_dist - neg_dist)**2).mean()
    if loss < torch.tensor(0, dtype=float) + 1e-8:
        return torch.tensor(0, requires_grad=True, dtype=float)
    else:
        return loss


def run_matching(amos, model, file_out, max_imgs=10, second_nearest=False, bsize=2000):
    model.eval()
    printc.green('processing patches ...')
    descs = get_descs(model, amos.patch_sets, bsize=bsize)  #descs.shape = (8489, 50, 256)
    precs = []
    # losses = np.zeros(descs.shape[:2])  #losses.shape = (8489, 50)
    counts = np.zeros(descs.shape[:2])

    printc.red('running standard version')
    gb = npi.group_by(amos.cam_idxs)
    all_idxs = gb.split_array_as_list(np.arange(len(amos.patch_sets)))
    printc.green('evaluating ...')

    APs = []
    for i, idxs in enumerate(all_idxs): #same as cam_idx (eg. 0, 1,2 ...33)
        desc = descs[idxs][:,:max_imgs] # subfolder10descs (for one cam)(desc.shape = (133, 10, 256))
        aux = np.arange(desc.shape[1])
        combs = list(itertools.permutations(aux, 2)) # includes (a,b), (b,a)
        # combs = list(itertools.combinations(aux, 2)) # only (a,b)
        rights = []
        q_centers = amos.data['LAFs'].data.cpu().numpy()[idxs][:,:,2] # 133patchcenter coordinates
        t_centers = amos.data['LAFs'].data.cpu().numpy()[idxs][:,:,2]# t_centers.shape = (171, 2)
        oneAP = []
        for c in tqdm(combs, desc='running pairs'): #（subfolder）10(（0，1）（0，2）...90
            colls = [np.array(amos.collisions[c])-idxs[0] for c in idxs] ### this should correct indices according to offset, we want idxs to current set
            # colls = [[list(idxs).index(c) for c in coll] for coll in colls] # maybe slow
            right, mask, AP = test_pair_(query=desc[:,c[0]], target=desc[:,c[1]], q_centers=q_centers, t_centers=t_centers, collisions=colls, second_nearest=second_nearest)
            oneAP += [AP]
            rights += [right]
            # losses[idxs,c[0]] += (1-rights[-1])     #losses is patches1
            counts[idxs,c[0]] += 1

        APs += oneAP
        precs += [np.mean(np.concatenate(rights))]
        print(amos.data['view_names'][i], 'correct rate= {:.2f}%'.format(precs[-1] * 100.0))    #amos.data['view_names'] is name of subfolder
        print(amos.data['view_names'][i], 'correct rate= {:.2f}%'.format(precs[-1] * 100.0), file=file_out)
        print(amos.data['view_names'][i], 'avg prec= {:.2f}%'.format(100.0 * np.mean(np.array(oneAP))))
        print(amos.data['view_names'][i], 'avg prec= {:.2f}%'.format(100.0 * np.mean(np.array(oneAP))), file=file_out)

    printc.green('mean correct rate={:.6f}%'.format(100.0*np.mean(np.array(precs))) )   #presc
    print('mean correct rate={:.6f}%'.format(100.0*np.mean(np.array(precs))), file=file_out)

    APs = np.array(APs)
    printc.green('mAP={:.6f}%'.format(100*np.mean(APs)) )
    print('mAP={:.6f}%'.format(100*np.mean(APs)), file=file_out)

    out = {}
    # out['losses'] = losses
    out['counts'] = counts
    out['data_path'] = amos.data_path
    out['type'] = 'matching'
    return out


def get_3_fcs(descs, cam_idxs, npts=100):
    all_idxs = list(np.arange(descs.shape[0]))
    ps_idxs = random.sample(all_idxs, npts)
    set_idxs = [random.choice(list(np.arange(descs.shape[1]))) for _ in range(npts)]

    gb = npi.group_by(list(cam_idxs))
    idxs_per_cam = [list(c) for c in gb.split_array_as_list(all_idxs)]
    a,b,c,d = [],[],[],[]
    ea,eb,ec = [],[],[]
    for ps_idx, set_idx in tqdm(zip(ps_idxs, set_idxs), desc='Running queries', total=npts):
        pos_idx = list(range(descs.shape[1]))
        pos_idx.remove(set_idx)
        pos_idx = random.choice(pos_idx)
        other_set = list(range(descs.shape[1]))
        other_set.remove(set_idx)
        other_set.remove(pos_idx)
        cam_idx = cam_idxs[list(np.arange(descs.shape[0])).index(ps_idx)]   #cam_idx=5 ps_idxcamidx
        in_cam_idxs = copy(idxs_per_cam[cam_idx])   #in_cam_idxs.shape = 601
        in_cam_idxs.remove(ps_idx)      #camidxsshape=600
        out_cam_idxs = list(set(all_idxs).difference(set(idxs_per_cam[cam_idx])))   #list out_cam_idxs.shape=1251
        #campatch idxs
        # other_idxs = list(np.arange(descs.shape[0]))
        # other_idxs.remove(ps_idx)

        query_desc = np.expand_dims(descs[ps_idx, set_idx], 0)      #shape=(1,128)
        descs_img = descs[in_cam_idxs, pos_idx] # is only one   shape=(600, 128)
        aux = descs[in_cam_idxs][other_set]     #descs[1:600][1:48] campatch descs（+patch ）
        descs_cam = np.reshape(aux, (-1, descs.shape[-1]))
        descs_other = np.reshape(descs[out_cam_idxs, :], (-1, descs.shape[-1])) # campatch descs
        descs_pos = np.expand_dims(descs[ps_idx, pos_idx], 0)   #shape=(1,128)

        a += [find_nearest(query_desc, descs_img)]
        b += [find_nearest(query_desc, descs_cam)]
        c += [find_nearest(query_desc, descs_other)]
        d += [find_nearest(query_desc, descs_pos)]      #patchdist

        ea += [d[-1]-a[-1]] #dist 
        eb += [d[-1]-b[-1]]
        ec += [d[-1]-c[-1]]
    return np.array((a,b,c,d)), np.array((ea,eb,ec))

def find_nearest(descs_query, descs_target):
    dists = spatial.distance.cdist(descs_query, descs_target, 'euclidean')
    return np.amin(dists, axis=1)[0]

def find_mean(descs_query, descs_target):
    dists = spatial.distance.cdist(descs_query, descs_target, 'euclidean')
    return np.mean(dists)

def fce(p):
    p = torch.from_numpy(p).float() # numpy -> tensor -> numpy, because pool.map on huge tensor would fail on "too many open files"
    return transformation(p).data.cpu().numpy()

def get_descs(model, patch_sets, bsize=2000):
    pool = multiprocessing.Pool(30)
    # torch.multiprocessing.set_sharing_strategy('file_system')
    if model.name == 'TwoSAE':
        patches_1 = patch_sets[:, 0, :, :, :]; patches_2 = patch_sets[:, 1, :, :, :]
        inputs_1 = list(tqdm(pool.imap(fce, patches_1.data.cpu().numpy()), total=len(patches_1), desc='Transforming patches'))
        inputs_2 = list(tqdm(pool.imap(fce, patches_2.data.cpu().numpy()), total=len(patches_2), desc='Transforming patches'))
        inputs_1 = torch.from_numpy(np.stack(inputs_1)).float(); inputs_2 = torch.from_numpy(np.stack(inputs_2)).float()
        sys.stdout.flush()
        idxs = np.arange(len(inputs_1))
        splits = np.array_split(idxs, max(1, (patches_1.shape[0] // bsize) ** 2))
    else:
        patches = patch_sets.view(-1, patch_sets.shape[-3], patch_sets.shape[-2], patch_sets.shape[-1])
        inputs = list(tqdm(pool.imap(fce, patches.data.cpu().numpy()), total=len(patches), desc='Transforming patches'))
        printc.green('stacking ...')
        inputs = torch.from_numpy(np.stack(inputs)).float()
        printc.green('finished')
        sys.stdout.flush()
        idxs = np.arange(len(inputs))
        splits = np.array_split(idxs, max(1, (patches.shape[0] // bsize) ** 2))     #spilt all patches according to batch size

    preds = []; preds_1 = []; preds_2 = []
    printc.green('finished')
    sys.stdout.flush()
    with torch.no_grad():
        for spl in tqdm(splits, desc='Getting descriptors'):
            if model.name == 'SAE':
                preds += [model(inputs[spl].cuda())[0].data.cpu().numpy()]
            elif model.name == 'TwoSAE':
                pred_1, pred_2 = model( (inputs_1[spl].cuda(), inputs_2[spl].cuda()) )[0]
                preds_1.append(pred_1.data.cpu().numpy()); preds_2.append(pred_2.data.cpu().numpy())
            elif model.name == 'TwoNet':
                preds += [model(inputs[spl].cuda())['final_desc'].data.cpu().numpy()]
            else:
                preds += [model(inputs[spl].cuda()).data.cpu().numpy()]

    if model.name == 'TwoSAE':
        preds_1 = np.concatenate(preds_1); preds_2 = np.concatenate(preds_2)
        preds = np.transpose( np.stack(( preds_1, preds_2), axis = 0), (1, 0, 2))
    # elif  model.name == 'TwoNet':
    #     return preds
    else:
        preds = np.concatenate(preds)
        preds = np.reshape(preds, (patch_sets.shape[0], patch_sets.shape[1], -1))
    return preds

def transform_patches(bsize, model, patch_sets):
    pool = multiprocessing.Pool(30)
    # torch.multiprocessing.set_sharing_strategy('file_system')
    if model.name == 'TwoSAE':
        patches_1 = patch_sets[:, 0, :, :, :];
        patches_2 = patch_sets[:, 1, :, :, :]
        inputs_1 = list(
            tqdm(pool.imap(fce, patches_1.data.cpu().numpy()), total=len(patches_1), desc='Transforming patches'))
        inputs_2 = list(
            tqdm(pool.imap(fce, patches_2.data.cpu().numpy()), total=len(patches_2), desc='Transforming patches'))
        inputs_1 = torch.from_numpy(np.stack(inputs_1)).float();
        inputs_2 = torch.from_numpy(np.stack(inputs_2)).float()
        sys.stdout.flush()
        idxs = np.arange(len(inputs_1))
        splits = np.array_split(idxs, max(1, (patches_1.shape[0] // bsize) ** 2))
        inputs = (inputs_1, inputs_2)
    else:
        patches = patch_sets.reshape(-1, patch_sets.shape[-3], patch_sets.shape[-2], patch_sets.shape[-1])
        inputs = list(tqdm(pool.imap(fce, patches.data.cpu().numpy()), total=len(patches), desc='Transforming patches'))
        printc.green('stacking ...')
        inputs = torch.from_numpy(np.stack(inputs)).float()
        printc.green('finished')
        sys.stdout.flush()
        idxs = np.arange(len(inputs))
        splits = np.array_split(idxs,
                                max(1, (patches.shape[0] // bsize) ** 2))  # spilt all patches according to batch size
    pool.close();pool.join()
    return inputs, splits

def get_descs_forTrain(inputs, splits, model, patch_sets):
    preds = []; preds_1 = []; preds_2 = []
    sys.stdout.flush()
    # with torch.no_grad():
    for spl in tqdm(splits, desc='Getting descriptors'):    #2116 loops, each loop 44 patches go through netmodel?
        if model.name == 'SAE':
            preds += [model(inputs[spl].cuda())[0]]
        elif model.name == 'TwoSAE':
            pred_1, pred_2 = model( (inputs_1[spl].cuda(), inputs_2[spl].cuda()) )[0]
            preds_1.append(pred_1.data.cpu().numpy()); preds_2.append(pred_2.data.cpu().numpy())
        elif model.name == 'TwoNet':
            preds += [model(inputs[spl].cuda())['final_desc']]
        else:
            preds += [model(inputs[spl].cuda())]

    if model.name == 'TwoSAE':
        preds_1 = torch.cat(preds_1); preds_2 = torch.cat(preds_2)
        preds = torch.transpose(torch.stack(( preds_1, preds_2), axis = 0), (1, 0, 2))
    else:
        preds = torch.cat(preds)
        preds = torch.reshape(preds, (patch_sets.shape[0], patch_sets.shape[1], -1))
        # preds = preds.view(patch_sets.shape[0], patch_sets.shape[1], -1)
    return preds


def get_avg_dist(descs, npts=100):
    all_idxs = list(np.arange(descs.shape[0]))
    Aps_idxs = random.sample(all_idxs, npts)
    Aset_idxs = [random.choice(list(np.arange(descs.shape[1]))) for _ in range(npts)]
    Bps_idxs = random.sample(all_idxs, npts)
    Bset_idxs = [random.choice(list(np.arange(descs.shape[1]))) for _ in range(npts)]
    obs = []
    for Aps_idx, Aset_idx, Bps_idx, Bset_idx in tqdm(zip(Aps_idxs, Aset_idxs, Bps_idxs, Bset_idxs), desc='Running queries', total=npts):
        query_desc = np.expand_dims(descs[Aps_idx, Aset_idx], 0)
        target_desc = np.expand_dims(descs[Bps_idx, Bset_idx], 0)
        obs += [find_nearest(query_desc, target_desc)]
    return obs

def get_amos(data_path, AMOS_RGB=False, depths='', only_D=False, transformation=trans_crop_resize):
    return AMOS_dataset(transform=transformation,
                        data_path=data_path,
                        Npositives=1,                #2 for TwoSAE
                        AMOS_RGB=AMOS_RGB,
                        depths=depths,
                        only_D=only_D,
                        use_collisions=True,       #True
                        random_tuple=False          #False for TwoSAE
                        )

def data_from_type(type):
    if type in ['AMOS-views-v4_pairs-match']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4-pairs/AMOS-views-v4-pairs_PS:0_WF:uniform_PG:meanImg_minsets:1000_masks:AMOS-masks.pt'
    elif type in ['AMOS-views-v4']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_PS:60000_WF:Hessian_PG:meanImg_masks:AMOS-masks.pt'
    elif type in ['AMOS-views-v4_hess_fair']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_PS:0_WF:Hessian_PG:meanImg_minsets:1000_masks:AMOS-masks.pt'
    elif type in ['AMOS-views-v4_uni']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_PS:60000_WF:uniform_PG:meanImg_masks:AMOS-masks.pt'
    elif type in ['AMOS-views-v4_uni_fair']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_PS:0_WF:uniform_PG:meanImg_minsets:1000_masks:AMOS-masks.pt'
    elif type in ['AMOS-views-v4_uni_fair_mini']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_PS:0_WF:uniform_PG:meanImg_minsets:100_masks:AMOS-masks.pt'
    elif type in ['AMOS-test-1']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1/AMOS-test-1_PS:0_WF:uniform_PG:meanImg_minsets:1000_masks:AMOS-masks.pt'
    elif type in ['AMOS-test-1-pairs']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1_pairs/AMOS-test-1_pairs_PS:0_WF:uniform_PG:meanImg_minsets:1000_masks:AMOS-masks.pt'
    elif type in ['AMOS-test-1-new']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1/AMOS-test-1_maxsets:1000_WF:Hessian_PG:new_masks:AMOS-masks.pt'


    elif type in ['sift']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1-downsized/AMOS-test-1-downsized_WF:Hessian_PG:sift_masks:AMOS-masks.pt'
    elif type in ['sift-split']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1-downsized-split/AMOS-test-1-downsized-split_WF:Hessian_PG:sift_masks:AMOS-masks.pt'
    elif type in ['sift-RGB']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1-downsized/AMOS-test-1-downsized_WF:Hessian_PG:sift_RGB_masks:AMOS-masks.pt'
    elif type in ['sift-D']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1-downsized/AMOS-test-1-downsized_WF:Hessian_PG:sift_RGB_depths_masks:AMOS-masks.pt'

    #============Myself:============
    elif type in ['Amos-train']:
        # data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_maxsets:2000_sigmas-v:v14_thr:0.00016_WF:Hessian_PG:new_depths_masks:AMOS-masks.pt'
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_maxsets:1501_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:None.pt'
    elif type in ['full1']:
        # data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_maxsets:2000_sigmas-v:v14_thr:0.00016_WF:Hessian_PG:new_depths_masks:AMOS-masks.pt'
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_maxsets:1500_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:None.pt'
    elif type in ['AMOS-test-1-Hessian_test']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4_testset-v4/AMOS-views-v4_testset-v4_maxsets:1501_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:None.pt'
    elif type in ['NIR_RGB_ref_testset']:
        data_path ='Datasets/NIR_RGB_ref/NIR_RGB_ref_testset/NIR_RGB_ref_testset_maxsets:1724_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:None.pt'
    elif type in ['NIR_RGB_ref0_trianset']:
        data_path ='Datasets/NIR_RGB_ref0/NIR_RGB_ref_train/NIR_RGB_ref_train_maxsets:2001_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:None.pt'
    elif type in ['NIR_RGB_ref_trianset']:
        data_path = 'Datasets/NIR_RGB_ref/NIR_RGB_ref_train/NIR_RGB_ref_train_maxsets:1724_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:None.pt'
    elif type in [  'test_anchor']:
        data_path = 'Datasets/NIR_RGB_ref/test_anchor/NIR_RGB_ref_testset_maxsets:1724_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:None.pt'

    else: assert False, 'invalid test type'
    return data_path

class Interface:

    def three_fcs(self,
                  model_name='HardNet8-Univ',
                  type='AMOS-test-1-Hessian_test',
                  ):
        model = load_hardnet(model_name)
        amos = get_amos(data_from_type(type))
        descs = get_descs(model, amos.patch_sets)
        r, e = get_3_fcs(descs, amos.cam_idxs.long(), 100)
        a = np.argsort(r[0])    #x，index()，
        r = [r[0][a], r[1][a], r[2][a], r[3][a]]    #rsort，
        a = np.argsort(e[0])
        e = [e[0][a], e[1][a], e[2][a]] #sort

        fig = plt.figure(figsize=(20, 10))
        plt.plot(r[0])
        plt.plot(r[1])
        plt.plot(r[2])
        # plt.plot(res[3], linestyle='dotted')
        plt.plot(r[3], 'o')
        plt.legend(['in image', 'in view', 'other views', 'positives'])
        plt.xlabel('point')
        plt.ylabel('distance')
        plt.title('distances')
        dir_out = os.path.join('Models', model_name, 'Graphs')
        os.makedirs(dir_out, exist_ok=True)
        fig.savefig(os.path.join(dir_out, '_'.join([type, 'dists.png'])), dpi=fig.dpi)

        fig = plt.figure(figsize=(20, 10))
        plt.plot(e[0])
        plt.plot(e[1])
        plt.plot(e[2])
        plt.legend(['in image', 'in view', 'other views'])
        plt.xlabel('point')
        plt.ylabel('edge')
        plt.title('edges')
        dir_out = os.path.join('Models', model_name, 'Graphs')
        os.makedirs(dir_out, exist_ok=True)
        fig.savefig(os.path.join(dir_out, '_'.join([type, 'edges.png'])), dpi=fig.dpi)
    def avg_dist(self,
                 # model_name='id:103_arch:h1_ds:v4_loss:tripletMargin_mpos:1.0_mneg:1.0_PS:60000_WF:uniform_PG:meanImg_tps:20000000_CamsB:5_masks_ep:1',
                 model_name='HardNet8-Univ',
                 # type='NIR_RGB_ref_testset',
                 type='AMOS-test-1-Hessian_test',
                 ):
        model = load_hardnet(model_name)
        amos = get_amos(data_from_type(type))
        descs = get_descs(model, amos.patch_sets)
        res = get_avg_dist(descs, 1000)
        print('avg dist:', np.mean(np.array(res)))
        print('min dist:', np.min(np.array(res)))
        print('max dist:', np.max(np.array(res)))

    def match(self,
              model_name='id:finetune_Test_B41',
              type= 'AMOS-test-1-Hessian_test',
              # type = 'NIR_RGB_ref0_trianset',
              # type = 'test_anchor',
              SN=False,
              only_D=False,
              bs=2000,
              use_fine_tune=False,
              ):
        printc.yellow('\n'.join(['Input arguments:'] + [str(x) for x in sorted(locals().items()) if x[0] != 'self']))
        model = load_hardnet(model_name)
        path_out = os.path.join('Models', model_name, 'Matching', type+'.txt')
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        if '64' not in model.name:
            transformation = trans_crop_resize
        else:
            transformation = trans_crop_resize64
        if type in ['sift-D']: # channels must be averaged
            amos = get_amos(data_from_type(type), AMOS_RGB=False, depths='dummy', only_D=only_D, transformation=transformation)
        else:
            amos = get_amos(data_from_type(type), AMOS_RGB=False, only_D=only_D,  transformation=transformation)

        if use_fine_tune:
            model2 = load_hardnet(model_name)
            out = fine_tune(amos, model, model2, open(path_out, 'w'), second_nearest=SN, bsize=bs)
            current_time = datetime.datetime.now()
            current_time = current_time.strftime("%m%d-%H_%M_%S")
            save_path = os.path.join('Models', model_name, 'Finetune_Time:{}_right:{}.pt'.format(current_time, out['mean correct rate']))
            printc.green('saving to: {} ...'.format(save_path))
            torch.save({'epoch': -1, 'state_dict': model.state_dict(),
                        'model_arch': model.name, 'create_time': current_time,
                        'mean correct rate': out['mean correct rate']}, save_path)
        else:
            out = run_matching(amos, model, open(path_out, 'w'), second_nearest=SN, bsize=bs)
            np.save(os.path.join('Models', model_name, 'Matching/info_{}_matching.npy'.format(type)), out)
            # out = run_test_useAnchorReg(amos, model, open(path_out, 'w'), second_nearest=SN, bsize=bs)
            # np.save(os.path.join('Models', model_name, 'Matching/info_{}_matching.npy'.format(type)), out)

    def FPR95_test(self,
                   model_name='HardNet8-Univ',
                   type='AMOS-test-1-Hessian_test',
                   ):
        model = load_hardnet(model_name)
        amos = get_amos(data_from_type(type))
        descs = get_descs(model, amos.patch_sets)
        # res = get_avg_dist(descs, 1000)


        print('avg dist:', np.mean(np.array(res)))
        print('min dist:', np.min(np.array(res)))
        print('max dist:', np.max(np.array(res)))

if __name__ == "__main__":
    I = Interface()
    I.match()
    # I.match(use_fine_tune=True)
    # I.three_fcs()
    # I.avg_dist()
#