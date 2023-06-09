# ====================================================================
# Adapted from code base found at: https://github.com/Lihui-Chen/FAWDN
#
# Uses adapted dataloaders, solvers, and pathing. Learn more at
# https://github.com/Paper99/SRFBN_CVPR19.
# ====================================================================

import argparse, time, os
import imageio

import options.options as option
from utils import util
from solvers import create_solver
import torch.utils.data



def inference(img_path):
    # use below when testing in command line.
    #parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    #parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    #opt = '/options/test/test_FAWDN_x2.json'  # path to model
    #opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = 2
    degrad = "BI"
    network_opt = {
        "which_model": "FAWDN",
        "num_features": 16,
        "in_channels": 3,
        "out_channels": 3,
        "num_steps": 2,
        "nDenselayer": 8,
        "nBlock": 8
    }
    model_name = "FAWDN"

    #if opt['self_ensemble']: model_name += 'plus'

    # create test dataloader
    bm_names = []
    test_loaders = []
    need_HR = False

    # loading dataset
    test_set = create_dataset(img_path)

    test_loader = create_dataloader(test_set)
    test_loaders.append(test_loader)
    print('===> Test Dataset: [%s]   Number of images: [%d]' % (test_set.name(), len(test_set)))
    bm_names.append(test_set.name())

    # create solver (and load model)
    solver = create_solver('sr')
    # Test phase
    print('===> Start Test')
    print("==================================================")
    print("Method: %s || Scale: %d || Degradation: %s"%(model_name, scale, degrad))

    for bm, test_loader in zip(bm_names, test_loaders):
        print("Test set : [%s]"%bm)

        sr_list = []
        path_list = []

        total_psnr = []
        total_ssim = []
        total_time = []

        for iter, batch in enumerate(test_loader):
            solver.feed_data(batch, need_HR=need_HR)

            # calculate forward time
            t0 = time.time()
            solver.test()
            t1 = time.time()
            total_time.append((t1 - t0))

            visuals = solver.get_current_visual(need_HR=need_HR)
            sr_list.append(visuals['SR'])

            #calculate PSNR/SSIM metrics on Python
            if need_HR:
                psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)
                total_psnr.append(psnr)
                total_ssim.append(ssim)
                path_list.append(os.path.basename(batch['HR_path'][0]).replace('HR', model_name))
                print("[%d/%d] %s || PSNR(dB)/SSIM: %.2f/%.4f || Timer: %.4f sec ." % (iter+1, len(test_loader),
                                                                                       os.path.basename(batch['LR_path'][0]),
                                                                                       psnr, ssim,
                                                                                        (t1 - t0)))
            else:
                path_list.append(os.path.basename(batch['LR_path'][0]))
                print("[%d/%d] %s || Timer: %.4f sec ." % (iter + 1, len(test_loader),
                                                       os.path.basename(batch['LR_path'][0]),
                                                       (t1 - t0)))

        if need_HR:
            print("---- Average PSNR(dB) /SSIM /Speed(s) for [%s] ----" % bm)
            print("PSNR: %.2f      SSIM: %.4f      Speed: %.4f" % (sum(total_psnr)/len(total_psnr),
                                                                  sum(total_ssim)/len(total_ssim),
                                                                  sum(total_time)/len(total_time)))
        else:
            print("---- Average Speed(s) for [%s] is %.4f sec ----" % (bm,
                                                                      sum(total_time)/len(total_time)))

        # save SR results for further evaluation on MATLAB
        #if need_HR:
        #    save_img_path = os.path.join('./results/SR/'+degrad, model_name, bm, "x%d"%scale)
        #else:
        save_img_path = os.path.join('/kaggle/working/'+bm, model_name, "x%d"%scale)

        print("===> Saving SR images of [%s]... Save Path: [%s]\n" % (bm, save_img_path))

        if not os.path.exists(save_img_path): os.makedirs(save_img_path)
        for img, name in zip(sr_list, path_list):
            imageio.imwrite(os.path.join(save_img_path, name), img)

    print("==================================================")
    print("===> Finished !")

def create_dataloader(dataset):
    #phase = dataset_opt['phase']
    #if phase == 'train':
    #    batch_size = dataset_opt['batch_size']
    #    shuffle = True
    #    num_workers = dataset_opt['n_workers']
    #else:
    batch_size = 1   # ?
    shuffle = False
    num_workers = 1  # ?
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def create_dataset(data_path):
    mode = "LR"                     # set if testing, "LRHR" if training and evaluating
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
        dataset = D(data_path)
    elif mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
        dataset = D(data_path,'/kaggle/working/128res/')
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % mode)
    # pass path
    print('===> [%s] Dataset is created.' % (mode))
    return dataset


if __name__ == '__main__':
    inference(imgs)
