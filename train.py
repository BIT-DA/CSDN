import argparse
import os
import time
import random
import sys
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

import src.network
import torch.optim as optim
import src.pre_process as prep
from src.pre_process import transforms
from src.data_list import ImageList
import src.lr_schedule
from src.logger import Logger
import numpy as np
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import warnings
import torchvision
from src.data_prep import DsetThreeChannels
from src.usps import USPS

cudnn.benchmark = True
cudnn.deterministic = True
warnings.filterwarnings('ignore')


def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def val(data_set_loader, base_net, f1_net, f2_net, test_10crop, config, num_iter):

    # base_net.train(False)
    # f1_net.train(False)
    # f2_net.train(False)

    base_net.eval()
    f1_net.eval()
    f2_net.eval()

    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(data_set_loader['test'][i]) for i in range(10)]
            for i in range(len(data_set_loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()

                outputs = []
                for j in range(10):
                    feature = base_net(inputs[j])
                    predict_out1 = f1_net(x=feature, alpha=0)
                    predict_out2 = f2_net(x=feature, alpha=0)
                    predict_out = predict_out1 + predict_out2
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)

                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), dim=0)
                    all_label = torch.cat((all_label, labels.float()), dim=0)
        else:
            iter_test = iter(data_set_loader['test'])
            for i in range(len(data_set_loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()

                feature = base_net(inputs)
                predict_out1 = f1_net(feature, 0)
                predict_out2 = f2_net(feature, 0)
                outputs = predict_out1 + predict_out2

                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # if config['is_writer']:
    #     config['writer'].add_scalars('test', {'test error': 1.0 - accuracy,
    #                                           'acc': accuracy * 100.0},
    #                                  num_iter)
    return accuracy * 100.0


def pred_target(data_set_loader, base_net, f1_net, f2_net, class_num, num_iter, max_iter):

    base_net.eval()
    f1_net.eval()
    f2_net.eval()

    start_pred = True
    with torch.no_grad():
        total = 0
        iter_target = iter(data_set_loader['target'])
        for i in range(len(data_set_loader['target'])):
            inputs, _ = iter_target.next()
            inputs = inputs.cuda()
            total += len(inputs)

            features = base_net(inputs)

            pred1 = f1_net(features)
            pred2 = f2_net(features)

            temperature = 2 - 1.5 * float(num_iter) / float(max_iter)
            pred1 = pred1 / temperature
            pred2 = pred2 / temperature

            pred1 = F.softmax(pred1)
            pred2 = F.softmax(pred2)

            if start_pred:
                pred = torch.sum(pred1 + pred2, dim=0)
                start_pred = False
            else:
                pred += torch.sum(pred1 + pred2, dim=0)
        pred = pred / (2 * total) * class_num
    return pred


def Domain_weight(d_net, feature_target):
    d_net.eval()
    instance_weight = d_net(feature_target)
    return instance_weight.view(instance_weight.size(0))

# 概率极端化 --梁
def prob2extreme(prob):
    scaled_prob = 2 * prob - 1
    extremed_prob=np.sin(scaled_prob*np.pi/2)
    return (extremed_prob+1)/2


def train(config):
    # set pre-process
    prep_dict = {'source': prep.image_train(**config['prep']['params']),
                 'target': prep.image_train(**config['prep']['params'])}
    if config['prep']['test_10crop']:
        prep_dict['test'] = prep.image_test_10crop(**config['prep']['params'])
    else:
        prep_dict['test'] = prep.image_test(**config['prep']['params'])

    # prepare data
    data_set = {}
    data_set_loader = {}
    data_config = config['data']

    path_mnist = '/media/ls/work/data/MNIST'
    path_svhn = '/media/ls/work/data/svhn'
    path_usps = '/media/ls/work/data/usps'
    image_size = 32
    if data_config['source']['list_path'] == 'svhn':
        data_set['source'] = torchvision.datasets.SVHN(root=path_svhn, split='test', download=True,
                                           transform=transforms.Compose([transforms.Resize(image_size),
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize((0.437, 0.4437, 0.4728),
                                                                                              (0.1980, 0.2010, 0.1970))
                                                                         ]))
        data_set_loader['source'] = torch.utils.data.DataLoader(data_set['source'],
                                                                batch_size=data_config['source']['batch_size'],
                                                                shuffle=True, num_workers=4)

    if data_config['source']['list_path'] == 'usps':
        data_set['source'] = USPS(path_usps , download=True, train=True, transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.2539,), (0.3842,))
        ]))
        data_set['source'] = DsetThreeChannels(data_set['source'])
        data_set_loader['source'] = torch.utils.data.DataLoader(data_set['source'],
                                                                batch_size=data_config['source']['batch_size'],
                                                                shuffle=True, num_workers=4)
    if data_config['source']['list_path'] == 'mnist':
        data_set['source'] = torchvision.datasets.MNIST(path_mnist + '/mnist', download=True, train=True,
                                                        transform=transforms.Compose([transforms.Resize(image_size),
                                                                                      transforms.ToTensor(),
                                                                                      transforms.Normalize((0.1307,),
                                                                                                           (0.3081,))
                                                                                      ]))
        data_set['source'] = DsetThreeChannels(data_set['source'])
        data_set_loader['source'] = torch.utils.data.DataLoader(data_set['source'],
                                                                batch_size=data_config['source']['batch_size'],
                                                                shuffle=True, num_workers=4)

    if data_config['target']['list_path'] == 'usps':
        data_set['target'] = USPS(path_usps , download=True, train=True, transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.2539,), (0.3842,))
        ]))
        data_set['target'] = DsetThreeChannels(data_set['target'])
        data_set_loader['target'] = torch.utils.data.DataLoader(data_set['target'],
                                                                batch_size=data_config['target']['batch_size'],
                                                                shuffle=True, num_workers=4)

        data_set['test'] = USPS(path_usps , download=True, train=True, transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.2539,), (0.3842,))
        ]))
        data_set['test'] = DsetThreeChannels(data_set['test'])
        data_set_loader['test'] = torch.utils.data.DataLoader(data_set['test'],
                                                              batch_size=data_config['test']['batch_size'],
                                                              shuffle=False, num_workers=4)

    if data_config['target']['list_path'] == 'mnist':
        print("mnist")
        data_set['target'] = torchvision.datasets.MNIST(path_mnist + '/mnist', download=True, train=False,
                                            transform=transforms.Compose([transforms.Resize(image_size),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,), (0.3081,))
                                                                          ]))
        data_set['target'] = DsetThreeChannels(data_set['target'])
        data_set_loader['target'] = torch.utils.data.DataLoader(data_set['target'],
                                                                batch_size=data_config['target']['batch_size'],
                                                                shuffle=True, num_workers=4)

        data_set['test'] = torchvision.datasets.MNIST(path_mnist + '/mnist', download=True, train=False,
                                            transform=transforms.Compose([transforms.Resize(image_size),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,), (0.3081,))
                                                                          ]))
        data_set['test'] = DsetThreeChannels(data_set['test'])
        data_set_loader['test'] = torch.utils.data.DataLoader(data_set['test'],
                                                              batch_size=data_config['test']['batch_size'],
                                                              shuffle=False, num_workers=4)

    elif config['data_set'] == 'office':
        data_set['source'] = ImageList(open(data_config['source']['list_path']).readlines(),
                                       transform=prep_dict['source'])
        data_set_loader['source'] = torch.utils.data.DataLoader(data_set['source'],
                                                                batch_size=data_config['source']['batch_size'],
                                                                shuffle=True, num_workers=4)
        data_set['target'] = ImageList(open(data_config['target']['list_path']).readlines(),
                                       transform=prep_dict['target'])
        data_set_loader['target'] = torch.utils.data.DataLoader(data_set['target'],
                                                                batch_size=data_config['target']['batch_size'],
                                                                shuffle=True, num_workers=4)
        if config['prep']['test_10crop']:
            data_set['test'] = [ImageList(open(data_config['test']['list_path']).readlines(),
                                          transform=prep_dict['test'][i])
                                for i in range(10)]
            data_set_loader['test'] = [torch.utils.data.DataLoader(dset, batch_size=data_config['test']['batch_size'],
                                                                   shuffle=False, num_workers=4)
                                       for dset in data_set['test']]
        else:
            data_set['test'] = ImageList(open(data_config['test']['list_path']).readlines(),
                                         transform=prep_dict['test'])
            data_set_loader['test'] = torch.utils.data.DataLoader(data_set['test'],
                                                                  batch_size=data_config['test']['batch_size'],
                                                                  shuffle=False, num_workers=4)

    # setup networks
    net_config = config['network']
    base_net = net_config['name'](use_bottleneck=config['use_bottleneck'])  # res50
    class_num = config['network']['params']['class_num']
    f1_net = src.network.ClassifierMCD(in_features=base_net.output_num(), hidden_size=1000, class_num=class_num)
    f2_net = src.network.ClassifierMCD(in_features=base_net.output_num(), hidden_size=1000, class_num=class_num)
    f1_net.init_weight()
    f2_net.init_weight()


    if config['pre_train']:
        checkpoint = torch.load('model/' + config['data']['source']['name'] + '.tar')
        base_net.load_state_dict(checkpoint['base'])
        # f1_net.load_state_dict(checkpoint['f1'])
        # f2_net.load_state_dict(checkpoint['f2'])

    base_net = base_net.cuda()
    f1_net = f1_net.cuda()
    f2_net = f2_net.cuda()

    # set optimizer
    optimizer_config = config['optimizer']
    schedule_param = optimizer_config['lr_param']
    lr_scheduler = src.lr_schedule.schedule_dict[optimizer_config['lr_type']]
    optimizer_g = optimizer_config['type'](base_net.get_parameters(), **(optimizer_config['optim_params']))
    optimizer_f = optimizer_config['type'](f1_net.get_parameters()+f2_net.get_parameters(),
                                           **(optimizer_config['optim_params']))
    # optimizer_d = optimizer_config['type'](d_net.get_parameters(), **(optimizer_config['optim_params']))

    # train
    len_train_source = len(data_set_loader['source'])
    len_train_target = len(data_set_loader['target'])
    best_acc = 0.0
    since = time.time()
    weighted = torch.ones(class_num)
    for num_iter in range(config['max_iter']):

        # update class-weight each 100 iterations
        if num_iter % config['update_iter'] == 0 and num_iter != 0:
            weighted = pred_target(data_set_loader, base_net, f1_net, f2_net, class_num, num_iter, config['max_iter'])

        if num_iter % config['val_iter'] == 0:
            # config['logger'].logger.debug(str(weighted))
            temp_acc = val(data_set_loader, base_net, f1_net, f2_net,
                           test_10crop=config['prep']['test_10crop'],
                           config=config, num_iter=num_iter)
            if temp_acc > best_acc:
                best_acc = temp_acc
            log_str = 'iter: {:d}, accu: {:.4f}\ntime: {:.4f}'.format(num_iter, temp_acc, time.time() - since)
            config['logger'].logger.debug(log_str)
            config['results'][num_iter].append(temp_acc)

        if num_iter % len_train_source == 0:
            iter_source = iter(data_set_loader['source'])
        if num_iter % len_train_target == 0:
            iter_target = iter(data_set_loader['target'])

        base_net.train(True)
        f1_net.train(True)
        f2_net.train(True)
        # d_net.train(True)

        # set loss
        class_criterion = nn.CrossEntropyLoss(weight=weighted.cuda())
        # class_criterion = nn.CrossEntropyLoss()
        # domain_criterion = nn.BCELoss()
        loss_params = config['loss']

        # set optimizer
        optimizer_g = lr_scheduler(optimizer_g, num_iter / config['max_iter'], **schedule_param)
        optimizer_g.zero_grad()
        optimizer_f = lr_scheduler(optimizer_f, num_iter / config['max_iter'], **schedule_param)
        optimizer_f.zero_grad()
        # optimizer_d = lr_scheduler(optimizer_d, num_iter / config['max_iter'], **schedule_param)
        # optimizer_d.zero_grad()

        input_source, label_source = iter_source.next()
        input_target, _ = iter_target.next()
        input_source, label_source, input_target = input_source.cuda(), label_source.cuda(), input_target.cuda()

        # Step 2: train G, F1, F2 to minimize loss on source
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        # optimizer_d.zero_grad()

        feature_source = base_net(input_source)
        feature_target = base_net(input_target)
        output_s1 = f1_net(feature_source)
        output_s2 = f2_net(feature_source)
        output_t1 = f1_net(feature_target)
        output_t2 = f2_net(feature_target)
        output_t1 = F.softmax(output_t1)
        output_t2 = F.softmax(output_t2)

        classifier_loss1 = class_criterion(output_s1, label_source)
        classifier_loss2 = class_criterion(output_s2, label_source)
        entropy_loss = EntropyLoss(output_t1) + EntropyLoss(output_t2)
        total_loss = classifier_loss1 + classifier_loss2 + config['entropy'] * entropy_loss
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        # Step 3: train classifiers to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        # optimizer_d.zero_grad()

        feature_source = base_net(input_source)
        feature_target = base_net(input_target)
        output_s1 = f1_net(feature_source)
        output_s2 = f2_net(feature_source)
        output_t1 = f1_net(feature_target)
        output_t2 = f2_net(feature_target)
        output_t1 = F.softmax(output_t1)
        output_t2 = F.softmax(output_t2)

        classifier_loss1 = class_criterion(output_s1, label_source)
        classifier_loss2 = class_criterion(output_s2, label_source)
        entropy_loss = EntropyLoss(output_t1) + EntropyLoss(output_t2)

        output_t1_norm = torch.sqrt(torch.sum(torch.pow(output_t1, 2), dim=1))
        output_t2_norm = torch.sqrt(torch.sum(torch.pow(output_t2, 2), dim=1))
        cos = torch.sum(torch.mul(output_t1, output_t2), dim=1) / torch.mul(output_t1_norm, output_t2_norm).detach()
        cos = (1 - cos) / torch.max(1 - cos)



        theta = torch.rand(config['num_theta'], output_t1.size(1)).cuda()
        theta = torch.mul(theta, 1 / torch.sum(theta, dim=1).unsqueeze(1))
        output_t1 = output_t1.transpose(1, 0)
        output_t2 = output_t2.transpose(1, 0)
        projection_t1 = torch.mm(theta, output_t1)
        projection_t2 = torch.mm(theta, output_t2)
        projection_t1, _ = torch.sort(projection_t1)
        projection_t2, _ = torch.sort(projection_t2)
        swd = torch.sum(torch.pow(projection_t1 - projection_t2, 2)) / config['num_theta']

        inconsistency_loss = swd
 

        total_loss = classifier_loss1 + classifier_loss2 - inconsistency_loss + config['entropy'] * entropy_loss
        total_loss.backward()
        optimizer_f.step()

        # Step 4 minimize discrepancy
        for i in range(config['num_k']):
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            # optimizer_d.zero_grad()

            feature_target = base_net(input_target)
            output_t1 = f1_net(feature_target)
            output_t2 = f2_net(feature_target)
            output_t1 = F.softmax(output_t1)
            output_t2 = F.softmax(output_t2)

            output_t1_norm = torch.sqrt(torch.sum(torch.pow(output_t1, 2), dim=1))
            output_t2_norm = torch.sqrt(torch.sum(torch.pow(output_t2, 2), dim=1))
            cos = torch.sum(torch.mul(output_t1, output_t2), dim=1) / torch.mul(output_t1_norm, output_t2_norm).detach()
            cos = (1 - cos) / torch.max(1 - cos)



            theta = torch.rand(config['num_theta'], output_t1.size(1)).cuda()
            theta = torch.mul(theta, 1 / torch.sum(theta, dim=1).unsqueeze(1))
            output_t1 = output_t1.transpose(1, 0)
            output_t2 = output_t2.transpose(1, 0)
            projection_t1 = torch.mm(theta, output_t1)
            projection_t2 = torch.mm(theta, output_t2)
            projection_t1, _ = torch.sort(projection_t1)
            projection_t2, _ = torch.sort(projection_t2)
            swd = torch.sum(torch.pow(projection_t1 - projection_t2, 2) * cos) / config['num_theta']
            inconsistency_loss = swd
            inconsistency_loss.backward()
            optimizer_g.step()

        if num_iter % config['val_iter'] == 0:
            print('class:', classifier_loss1.item() + classifier_loss2.item(),
                  'inconsistency:', inconsistency_loss.item() * loss_params['dis_off'],
                  'total:', total_loss.item())
            if config['is_writer']:
                config['writer'].add_scalars('train', {'total': total_loss.item(), 'class': classifier_loss1.item() + classifier_loss2.item(),
                                                       'inconsistency': inconsistency_loss.item() * loss_params[
                                                           'dis_off']},
                                             num_iter)

    if config['is_writer']:
        config['writer'].close()

    return best_acc


def empty_dict(config):
    config['results'] = {}
    for i in range(config['max_iter'] // config['val_iter'] + 1):
        key = config['val_iter'] * i
        config['results'][key] = []
    config['results']['best'] = []


def print_dict(config):
    for i in range(config['max_iter'] // config['val_iter'] + 1):
        key = config['val_iter'] * i
        log_str = 'iter: {:d}, average: {:.4f}'.format(key, np.average(config['results'][key]))
        config['logger'].logger.debug(log_str)
    log_str = 'best, average: {:.4f}'.format(np.average(config['results']['best']))
    config['logger'].logger.debug(log_str)
    config['logger'].logger.debug('-' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint Adversarial Domain Adaptation')
    parser.add_argument('--seed', type=int, default=1, help='manual seed')
    parser.add_argument('--gpu', type=str, nargs='?', default='3', help='device id to run')
    parser.add_argument('--net', type=str, default='ResNet50', help='Options: ResNet18,34,50,101,152;AlexNet')
    parser.add_argument('--data_set', type=str, default='office', help='Options: office,clef,digital,visda,home')
    parser.add_argument('--source_path', type=str, default='../amazon_list.txt', help='The source list')
    parser.add_argument('--target_path', type=str, default='../webcam10_list.txt', help='The target list')
    parser.add_argument('--max_iter', type=int, default=10001, help='max iterations')
    parser.add_argument('--val_iter', type=int, default=500, help='interval of two continuous test phase')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate') 
    parser.add_argument('--gamma', type=float, default=0.0001, help='learning rate')    # 0.001
    parser.add_argument('--batch_size', type=int, default=36, help='mini batch size')
    parser.add_argument('--output_path', type=str, default='output/', help='save log/scalar/model file path')
    parser.add_argument('--log_file', type=str, default='office31', help='log file name')
    parser.add_argument('--is_writer', type=bool, default=False, help='whether record for tensorboard')
    parser.add_argument('--update_iter', type=int, default=500, help='interval of two continuous output model')
    parser.add_argument('--pre_train', type=bool, default=False, help='use pre_train model')
    parser.add_argument('--use_bottleneck', type=bool, default=True, help='use bottleneck layer')
    parser.add_argument('--num_k', type=int, default=1, help='how many times G update ')
    parser.add_argument('--num_theta', type=int, default=256, help='number of theta samples')
    parser.add_argument('--entropy', type=float, default=0.1)
    args = parser.parse_args()

    # seed for everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONASHSEED'] = str(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = {'seed': args.seed, 'gpu': args.gpu, 'max_iter': args.max_iter, 'val_iter': args.val_iter,
              'is_writer': args.is_writer, 'update_iter': args.update_iter, 'num_k':args.num_k,
              'pre_train': args.pre_train, 'use_bottleneck': args.use_bottleneck,
              'output_path': args.output_path + args.data_set, 'loss': {'domain_off': 1.0, 'dis_off': 1.0},
              'prep': {'test_10crop': True, 'params': {'resize_size': 256, 'crop_size': 224}},
              'network': {'name': src.network.ResBase, 'params': {'resnet_name': args.net}},
              'optimizer': {'type': optim.SGD,
                            'optim_params': {'lr': args.lr, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
                            'lr_type': 'inv',
                            'lr_param': {'lr': args.lr, 'gamma': args.gamma, 'power': 0.75}}, 'data_set': args.data_set,
              'data': {
                  'source': {'name': args.source_path.split('/')[-1].split('_')[0], 'list_path': args.source_path,
                             'batch_size': args.batch_size},
                  'target': {'name': args.target_path.split('/')[-1].split('_')[0], 'list_path': args.target_path,
                             'batch_size': args.batch_size},
                  'test': {'list_path': args.target_path, 'batch_size': args.batch_size}},
              'num_theta': args.num_theta,
              'entropy': args.entropy
              }

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
    if config['is_writer']:
        config['writer'] = SummaryWriter(log_dir=config['output_path'] + '/scalar/', )
    config['logger'] = Logger(logroot=config['output_path'], filename=args.log_file, level='debug')

    empty_dict(config)
    if config['data_set'] == 'office':
        config['network']['params']['class_num'] = 31
    elif config['data_set'] == 'clef':
        config['network']['params']['class_num'] = 12
    elif config['data_set'] == 'digit':
        config['network']['params']['class_num'] = 10

  
    import pprint
    config['data']['source']['list_path'] = args.source_path
    config['data']['target']['list_path'] = args.target_path
    config['data']['test']['list_path'] = args.target_path
    config['logger'].logger.debug(pprint.pformat(config))
    config['logger'].logger.debug(
        config['data']['source']['list_path'] + ' vs ' + config['data']['target']['list_path'] +
        ' lr: ' + str(config['optimizer']['lr_param']['lr']) + ' num_theta: ' + str(
            config['num_theta']))
    config['logger'].logger.debug(time.time())
    empty_dict(config)
    config['results']['best'].append(train(config))
    print_dict(config)
