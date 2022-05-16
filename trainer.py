'''
trainer.py

Train 3dgan models
'''

import torch
from torch import optim
from torch import nn
from utils import *
import os

from model import net_G, net_D

# added
import datetime
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import params
from tqdm import tqdm


def save_train_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['train_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['train_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def save_val_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['val_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['val_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def trainer(args):
    # added for output dir
    save_file_path = params.output_dir + '/' + args.model_name
    print(save_file_path)  # ../outputs/dcgan
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    # for using tensorboard
    if args.logs:
        model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        # writer = SummaryWriter(params.output_dir+'/'+args.model_name+'/'+model_uid+'_'+args.logs+'/logs')
        writer = SummaryWriter(params.output_dir + '/' + args.model_name + '/' + args.logs + '/logs')

        image_saved_path = params.output_dir + '/' + args.model_name + '/' + args.logs + '/images'
        model_saved_path = params.output_dir + '/' + args.model_name + '/' + args.logs + '/models'

        if not os.path.exists(image_saved_path):
            os.makedirs(image_saved_path)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)

    # datset define
#     dsets_path = params.data_dir + params.model_dir + "train/"
    dsets_path = params.data_dir + params.model_dir + "30/train/"
    # if params.cube_len == 64:
    #     dsets_path = params.data_dir + params.model_dir + "30/train64/"

    print(dsets_path)  # ../volumetric_data/chair/30/train/

    train_dsets = ShapeNetDataset(dsets_path, args, "train")
    # val_dsets = ShapeNetDataset(dsets_path, args, "val")
    if torch.cuda.is_available():

        train_dset_loaders = torch.utils.data.DataLoader(train_dsets, batch_size=params.batch_size*torch.cuda.device_count(), shuffle=True,
                                                         num_workers=2)
    else:
        train_dset_loaders = torch.utils.data.DataLoader(train_dsets, batch_size=params.batch_size, shuffle=True,
                                                         num_workers=2)
    # val_dset_loaders = torch.utils.data.DataLoader(val_dsets, batch_size=args.batch_size, shuffle=True, num_workers=1)

    dset_len = {"train": len(train_dsets)}
    dset_loaders = {"train": train_dset_loaders}
    # print (dset_len["train"])

    # model define
    D = net_D(args)
    G = net_G(args)
#     model = Model(input_size, output_size)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        D = nn.DataParallel(D)
        G = nn.DataParallel(G)

    # print total number of parameters in a model
    # x = sum(p.numel() for p in G.parameters() if p.requires_grad)
    # print (x)
    # x = sum(p.numel() for p in D.parameters() if p.requires_grad)
    # print (x)
    D.to(params.device)
    G.to(params.device)

    if params.hyper_set==7:
        D_solver = optim.SGD(D.parameters(), lr=params.d_lr, momentum=0.9)
        # D_solver = optim.SGD(D.parameters(), lr=params.d_lr)
    #     D_solver = optim.Adadelta(D.parameters(), lr=params.d_lr)
        # G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)
        G_solver = optim.SGD(G.parameters(), lr=params.g_lr, momentum=0.9)
        # G_solver = optim.SGD(G.parameters(), lr=params.g_lr)
    elif params.hyper_set==4:
        D_solver = optim.Adadelta(D.parameters(), lr=params.d_lr)
    #     D_solver = optim.Adadelta(D.parameters(), lr=params.d_lr)
        G_solver = optim.Adadelta(G.parameters(), lr=params.g_lr)
    else:
        D_solver = optim.Adam(D.parameters(), lr=params.d_lr, betas=params.beta)
    #     D_solver = optim.Adadelta(D.parameters(), lr=params.d_lr)
        G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)


    criterion_D = nn.MSELoss()
    criterion_G = nn.L1Loss()

    itr_val = -1
    itr_train = -1

    ##gathering losses for plotting
    generator_recon_losses=[]
    generator_adv_losses=[]
    descriminator_real_losses=[]
    descriminator_fake_losses=[]
    per_epoch_time=[]


    for epoch in range(params.epochs):

        start = time.time()

        for phase in ['train']:
            if phase == 'train':
                # if args.lrsh:
                #     D_scheduler.step()
                D.train()
                G.train()
            else:
                D.eval()
                G.eval()

            running_loss_G = 0.0
            running_loss_D = 0.0
            running_loss_adv_G = 0.0

            for i, X in enumerate(tqdm(dset_loaders[phase])):

                # if phase == 'val':
                #     itr_val += 1

                if phase == 'train':
                    itr_train += 1

                X = X.to(params.device)
                # print (X)
                # print (X.size())

                batch = X.size()[0]
                # print (batch)

                Z = generateZ(args, batch)
                # print ('generator input shape',Z.size())

                # ============= Train the discriminator =============#
                # print ('Discriminator input shape',X.size())
                d_real = D(X)

                fake = G(Z)
                d_fake = D(fake)

                real_labels = torch.ones_like(d_real).to(params.device)
                fake_labels = torch.zeros_like(d_fake).to(params.device)
                # print (d_fake.size(), fake_labels.size())

                if params.soft_label:
                    real_labels = torch.Tensor(batch).uniform_(0.7, 1.2).to(params.device)
                    fake_labels = torch.Tensor(batch).uniform_(0, 0.3).to(params.device)

                # print (d_real.size(), real_labels.size())
                d_real_loss = criterion_D(d_real, real_labels)

                d_fake_loss = criterion_D(d_fake, fake_labels)

                d_loss = d_real_loss + d_fake_loss

                # no deleted
                d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
                d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
                d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

                if d_total_acu < params.d_thresh:
                    D.zero_grad()
                    d_loss.backward()
                    D_solver.step()

                # =============== Train the generator ===============#

                Z = generateZ(args, batch)

                # print (X)
                fake = G(Z)  # generated fake: 0-1, X: 0/1
                d_fake = D(fake)

                adv_g_loss = criterion_D(d_fake, real_labels)
                # print (fake.size(), X.size())

                # recon_g_loss = criterion_D(fake, X)
                recon_g_loss = criterion_G(fake, X)
                # g_loss = recon_g_loss + params.adv_weight * adv_g_loss
                g_loss = adv_g_loss

                if args.local_test:
                    # print('Iteration-{} , D(x) : {:.4} , G(x) : {:.4} , D(G(x)) : {:.4}'.format(itr_train, d_loss.item(), recon_g_loss.item(), adv_g_loss.item()))
                    print('Iteration-{} , D(x) : {:.4}, D(G(x)) : {:.4}'.format(itr_train, d_loss.item(),
                                                                                adv_g_loss.item()))

                D.zero_grad()
                G.zero_grad()
                g_loss.backward()
                G_solver.step()

                # =============== logging each 10 iterations ===============#

                running_loss_G += recon_g_loss.item() * X.size(0)
                running_loss_D += d_loss.item() * X.size(0)
                running_loss_adv_G += adv_g_loss.item() * X.size(0)




                if args.logs:
                    loss_G = {
                        'adv_loss_G': adv_g_loss,
                        'recon_loss_G': recon_g_loss,
                    }

                    loss_D = {
                        'adv_real_loss_D': d_real_loss,
                        'adv_fake_loss_D': d_fake_loss,
                    }

                    # if itr_val % 10 == 0 and phase == 'val':
                    #     save_val_log(writer, loss_D, loss_G, itr_val)

                    if itr_train % 10 == 0 and phase == 'train':
                        save_train_log(writer, loss_D, loss_G, itr_train)

            # =============== each epoch save model or save image ===============#
            epoch_loss_G = running_loss_G / dset_len[phase]
            epoch_loss_D = running_loss_D / dset_len[phase]
            epoch_loss_adv_G = running_loss_adv_G / dset_len[phase]

            #appending losses for each data
            generator_recon_losses.append(recon_g_loss.cpu().detach().numpy())
            generator_adv_losses.append(adv_g_loss.cpu().detach().numpy())
            descriminator_real_losses.append(d_real_loss.cpu().detach().numpy())
            descriminator_fake_losses.append(d_fake_loss.cpu().detach().numpy())

            end = time.time()
            epoch_time = end - start
            per_epoch_time.append(epoch_time)
#             print('time takes by one epoch kp',epoch_time)
            print('Epochs-{} ({}) , D(x) : {:.4}, D(G(x)) : {:.4}'.format(epoch, phase, epoch_loss_D, epoch_loss_adv_G))
            print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))

            if (epoch + 1) % params.model_save_step == 0:
                print('model_saved, images_saved...')
                torch.save(G.state_dict(), model_saved_path + '/G.pth')
                torch.save(D.state_dict(), model_saved_path + '/D.pth')

                samples = fake.cpu().data[:8].squeeze().numpy()
                # print (samples.shape)
                # image_saved_path = '../images'

                SavePloat_Voxels(samples, image_saved_path, epoch)

    #saving loss graphs as image
#     plot1=plt()
    plt.plot(generator_recon_losses)
    plt.show()
    plt.savefig(params.output_dir+'/generator_recon_losses.png')
    plt.close()
#     plot2=plt()
    plt.plot(generator_adv_losses)
    plt.show()
    plt.savefig(params.output_dir+'/generator_adv_losses.png')
    plt.close()
#     plot3=plt()
    plt.plot(descriminator_real_losses)
    plt.show()
    plt.savefig(params.output_dir+'/descriminator_real_losses.png')
    plt.close()
#     plot4=plt()
    plt.plot(descriminator_fake_losses)
    plt.show()
    plt.savefig(params.output_dir+'/descriminator_fake_losses.png')
    plt.close()
