import torch
import numpy as np
import argparse
import os
import pickle
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion
from model_res_fusion import CNN_TSA_MLT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from torch.optim.lr_scheduler import MultiStepLR
import video_reader
import random
import logging
from utils import device
import datetime

def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# logger for training accuracies
train_logger = setup_logger('Training_accuracy', 'runs_trms/train_output.log')
# logger for evaluation accuracies
eval_logger = setup_logger('Evaluation_accuracy', 'runs_trms/eval_output.log')

###############################################3#
#增加随机数种子
# setting up seeds
manualSeed = random.randint(1, 10000)
manualSeed = 5194
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
################################################3
"""
Command line parser
"""
def parse_command_line():
    parser = argparse.ArgumentParser()
    # 经常需要修改
    parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf"], default="hmdb",
                        help="Dataset to use.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--tasks_per_batch", type=int, default=16,
                        help="Number of tasks between parameter optimizations.")
    parser.add_argument("--checkpoint_dir", "-c", default="./checkpoint_hmdb/", help="Directory to save checkpoint to.")
    # 经常需要修改
    parser.add_argument("--test_checkpoint_iter", type=int, default="5000", help="Path to model to load and test.")
    parser.add_argument("--test_model_path", "-m", default="./checkpoint_hmdb/checkpoint.pt",  help="Path to model to load and test.")
    # 经常需要修改
    parser.add_argument("--training_iterations", "-i", type=int, default=15020, help="Number of meta-training iterations.")
    # 经常需要修改
    parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
    parser.add_argument('--test_iters', nargs='+', type=int, default=[5000, 10000,15000],   #100020    #在训练的第几次开始进行测试
                        help='iterations to test at. Default is for ssv2 otam split.')
    # 经常须有修改
    parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint",  default= False,  action="store_true", help="Restart from latest checkpoint.")

    parser.add_argument("--way", type=int, default=5, help="Way of each task.")
    parser.add_argument("--shot", type=int, default= 5, help="Shots per class.")
    parser.add_argument("--query_per_class", type=int, default=5,
                        help="Target samples (i.e. queries) per class used for training.")
    parser.add_argument("--query_per_class_test", type=int, default=1,
                        help="Target samples (i.e. queries) per class used for testing.")
    parser.add_argument("--print_freq", type=int, default=100, help="print and log every n iterations.")  # 1000
    parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
    parser.add_argument("--num_workers", type=int, default=10, help="Num dataloader workers.")
    parser.add_argument("--method", choices=["resnet18", "resnet34", "resnet50"], default="resnet50", help="method")
    parser.add_argument("--trans_linear_out_dim", type=int, default=1152, help="Transformer linear_out_dim")
    parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
    parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
    #可能需要修改
    parser.add_argument("--save_freq", type=int, default=1000,  help="Number of iterations between checkpoint saves.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
    # 经常需要修改  default=[2, 3，4，5]  #修改为2个帧的组合
    parser.add_argument('--temp_set', nargs='+', type=int, help='cardinalities e.g. 2,3 is pairs and triples',   default=[2,3])
    parser.add_argument("--scratch", choices=["bc", "bp", "new"], default="bp",  help="directory containing dataset, splits, and checkpoint saves.")
    # 经常需要修改
    parser.add_argument("--gpus_use", default=[1,2,3], help="GPUs No. to split the ResNet over")
    parser.add_argument("--num_gpus", type=int, default=3, help="Number of GPUs to split the ResNet over")

    parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")
    #可能需要修改 不同的split这里的编号不一样
    parser.add_argument("--split", type=int, default=3, help="Dataset sp lit.")
    parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])

    # 经常需要修改
    parser.add_argument("--test_model_only", type=bool, default=False, help="Only testing the model from the given checkpoint")

    parser.add_argument("--combination_number_used", type=int, default=17, help="combination_number_used")
    parser.add_argument("--temporal_set", type=int, default=[1, 2, 3, 4], help="temporal_set")
    parser.add_argument("--level_number_set", type=int, default=[8, 4, 3, 2], help="temporal_set")
    '''
    temporal_set: is the array of the feature level, for example [1,2,3,4] means we sue the level 1, leve 2, leve 3, leve 4
    level_number_set: is the number of combination under each lever we use.
    the temporal_set and level_number_set shoud map to each other.
    '''
    parser.add_argument("--lambdas", type=int, default=[1, 0, 0, 0], help="lambdas")

    args = parser.parse_args()
    # 数据集存储在不同的地方 移动硬盘，服务器，本地PC
    if args.scratch == "bc":
        args.num_gpus = 2
        args.scratch = "./"
        args.scratch_data = "/mnt/mydata/mydata/video-mini-frames/"
    elif args.scratch == "bp":
        args.num_gpus = 3
        # this is low becuase of RAM constraints for the data loader
        args.num_workers = 3
        args.scratch = "./"
        args.scratch_data = "/home/guofei/mydata/video-mini-frames/"
    elif args.scratch == "new":
        args.num_workers = 2
        args.num_gpus = 1
        args.scratch = "./"
        args.scratch_data = "E:/mydata/video-mini-frames/"

    if args.checkpoint_dir == None:
        print("need to specify a checkpoint dir")
        exit(1)

    if (args.method == "resnet50") or (args.method == "resnet34"):
        args.img_size = 224

    if args.method == "resnet50":
        args.trans_linear_in_dim = 2048
    else:
        args.trans_linear_in_dim = 512

    if args.dataset == "ssv2":
        args.traintestlist = os.path.join(args.scratch, "splits/ssv2_OTAM")
        args.path = os.path.join(args.scratch_data, "mini-ssv2-frames-number/")
    elif args.dataset == "kinetics":
        args.traintestlist = os.path.join(args.scratch, "splits/kinetics_CMN")
        args.path = os.path.join(args.scratch_data, "kinetics-mini-frames/")
    elif args.dataset == "ucf":
        args.traintestlist = os.path.join(args.scratch, "splits/ucf_ARN")
        args.path = os.path.join(args.scratch_data, "UCF101_frames/")
    elif args.dataset == "hmdb":
        args.traintestlist = os.path.join(args.scratch, "splits/hmdb_ARN")
        args.path = os.path.join(args.scratch_data, "HMDB_51/")
    elif args.dataset == "ssv2_cmn":
        args.traintestlist = os.path.join(args.scratch, "splits/ssv2_CMN")
        args.path = os.path.join(args.scratch_data, "mini-ssv2-frames/")
    with open("args.pkl", "wb") as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
    return args
##################################################

def main():
    learner = Learner()
    learner.run()

class Learner:

    def __init__(self):
        self.args = parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        #默认的硬件
        self.device = device

        #初始化主模型
        self.model = self.init_model()

        #这几个变量似乎没有用
        ################################################################################
        self.train_set, self.validation_set, self.test_set = self.init_data()
        ################################################################################

        self.vd = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.args.num_workers)
        # 进行loss function 以及accurary_fn的定义，这两个方法定义在untils.py中
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        # 根据参数定义优化器
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)

        self.test_accuracies = TestAccuracies(self.test_set)
        # 其作用在于对optimizer中的学习率进行更新、调整，更新的方法是scheduler.step()。
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)

        self.start_iteration = 0
        # 根据参数，决定是否从checkpoint中加载模型
        if self.args.resume_from_checkpoint:
            self.load_checkpoint(0)
        self.optimizer.step()
        # 用0梯度，初始化优化器
        self.optimizer.zero_grad()

    def init_model(self):
        model = CNN_TSA_MLT(self.args)
        model = model.to(self.device)
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        return train_set, validation_set, test_set


    def run(self):
        # 训练精确度
        train_accuracies = []
        # 损失
        losses = []
        #总的迭代次数
        total_iterations = self.args.training_iterations
        #开始迭代次数
        iteration = self.start_iteration

        if self.args.test_model_only:
            print("Model being tested at path: " + self.args.test_model_path)
            self.load_checkpoint(self.args.test_checkpoint_iter)
            accuracy_dict = self.test(1)
            print(accuracy_dict)
            return

        ################################################################################
        #训练每次循环就是一个episode
        for task_dict in self.video_loader:
            if iteration >= total_iterations:
                break
            iteration += 1
            torch.set_grad_enabled(True)
            print("start to train iteration: " + str(iteration))
            start = datetime.datetime.now()
            task_loss, task_accuracy = self.train_task(task_dict)
            end = datetime.datetime.now()
            print("AAAAAAAAAAAAAAA->:" + str(end-start))
            print("finish to train iteration: " + str(iteration))
            train_accuracies.append(task_accuracy)
            losses.append(task_loss)

            # optimize
            #每tasks_per_batch次iteration进行一次网络参数更新, 模型开始新一轮的迭代
            if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()
            #每print_freq个iteration会打印一次日志
            #每print_freq个iteration计算一次平均loss和平均的training accuracy
            if (iteration + 1) % self.args.print_freq == 0:
                # print training stats
                # console log
                print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                              .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                      torch.Tensor(train_accuracies).mean().item()))

                #log file
                train_logger.info("For Task: {0}, the training loss is {1} and Training Accuracy is {2}".format(iteration + 1,
                                                                                                  torch.Tensor(losses).mean().item(),
                                                                                                  torch.Tensor(train_accuracies).mean().item()))

                avg_train_acc = torch.Tensor(train_accuracies).mean().item()
                avg_train_loss = torch.Tensor(losses).mean().item()

                train_accuracies = []
                losses = []
            #每save_freq个episode保存一次模型，最有一个episode不进行保存，而直接在后面使用torch.save进行保存
            if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                self.save_checkpoint(iteration + 1)

            # 这个是说test_iters个episode之后进行一次测试，有就是用正在训练的模型进行一次测试
            if ((iteration + 1) in self.args.test_iters) and (iteration + 1) != total_iterations:
                accuracy_dict = self.test(iteration + 1)
                print(accuracy_dict)
                self.test_accuracies.print(self.logfile, accuracy_dict)

        # save the final model
        torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.logfile.close()
   #task_dict['support_set']-->1 200 3 224 224
    def train_task(self, task_dict):
        #输入是task_dict 由video_reader读取得到的对象， 输出 support样本, query样本, support labels, query labels,  real_target_labels
        context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)
        #跑模型
        model_dict = self.model(context_images, context_labels, target_images)

        lambdas = self.args.lambdas
        target_logits = model_dict['l2_logits']
        target_ot_logits = model_dict['ot_logits']
        target_fusion_logits = model_dict['fusion_logits']

        task_logits_total = lambdas[0] * target_logits + lambdas[1] * target_ot_logits + lambdas[2]*target_fusion_logits
        task_accuracy = self.accuracy_fn(task_logits_total, target_labels)

        task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch
        task_ot_loss = self.loss(target_ot_logits, target_labels, self.device) / self.args.tasks_per_batch
        task_fusion_loss = self.loss(target_fusion_logits, target_labels, self.device) / self.args.tasks_per_batch

        task_loss_total = lambdas[0] * task_loss + lambdas[1] * task_ot_loss +  lambdas[2] * task_fusion_loss

        task_loss_total.backward(retain_graph=False)

        return task_loss_total, task_accuracy

    def test(self, num_episode):
        #在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
        self.model.eval()
        with torch.no_grad():
                # 此参数为False,则video_loader加载的是测试数据集
                self.video_loader.dataset.train = False
                accuracy_dict ={}
                accuracies = []
                losses = []
                iteration = 0

                tmp_accuracies = []
                tmp_losses = []
                # 数据集的名称
                item = self.args.dataset
                for task_dict in self.video_loader:
                    # num_test_tasks为测试中总的episode数量
                    if iteration >= self.args.num_test_tasks:
                        break
                    iteration += 1

                    context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)
                    model_dict = self.model(context_images, context_labels, target_images)

                    lambdas = self.args.lambdas
                    target_logits = model_dict['l2_logits']
                    target_ot_logits = model_dict['ot_logits']
                    target_fusion_logits = model_dict['fusion_logits']

                    target_logits_total = lambdas[0] * target_logits + lambdas[1] * target_ot_logits + lambdas[2] * target_fusion_logits

                    accuracy = self.accuracy_fn(target_logits_total, target_labels)
                    task_loss = self.loss(target_logits_total, target_labels, self.device) / self.args.tasks_per_batch
                    task_ot_loss = self.loss(target_ot_logits, target_labels, self.device) / self.args.tasks_per_batch
                    task_fusion_loss = self.loss(target_fusion_logits, target_labels, self.device) / self.args.tasks_per_batch

                    task_loss_total = lambdas[0] * task_loss + lambdas[1] * task_ot_loss + lambdas[2] * task_fusion_loss

                    eval_logger.info(
                        "For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(iteration + 1,
                                                                                                    task_loss_total.item(),
                                                                                                    accuracy.item()))
                    print("For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(iteration + 1,
                                                                                                      task_loss_total.item(),
                                                                                                      accuracy.item()))
                    losses.append(task_loss_total.item())
                    accuracies.append(accuracy.item())

                    ###########################################################################################################
                    '''以下代码块的内容是为了进行一个较小集合的计算 作为参考'''
                    tmp_losses.append(task_loss_total.item())
                    tmp_accuracies.append(accuracy.item())
                    if (iteration-1) % 100 == 0 and iteration > 1:
                        tmp_accuracy = np.array(tmp_accuracies).mean() * 100.0
                        # 总体标准差 等于 样本标准差除以根号下样本数量n
                        tmp_confidence = (196.0 * np.array(tmp_accuracies).std()) / np.sqrt(len(tmp_accuracies))
                        tmp_loss = np.array(losses).mean()  # loss取均值
                        accuracy_dict[item] = {"accuracy": tmp_accuracy, "confidence": tmp_confidence, "loss": tmp_loss}
                        print("#####################   The databse is {}, and the iteration is {}   ######################".format( item, num_episode))
                        print(accuracy_dict)
                        print( "##############################################################################################")
                        tmp_accuracies = []
                        tmp_losses = []
                    #############################################################################################################
                    del target_logits
                    del target_ot_logits
                    #del target_sim_kl_logits
                    #del target_part_ot_logits

                accuracy = np.array(accuracies).mean() * 100.0
                # 总体标准差 等于 样本标准差除以根号下样本数量n
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
                loss = np.array(losses).mean()  # loss取均值
                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence, "loss": loss}
                eval_logger.info("#####################   The databse is {}, and the iteration is {}   ######################".format(item, num_episode))
                eval_logger.info(accuracy_dict)
                eval_logger.info("##############################################################################################")
                eval_logger.info("----------------------------------------------------------------------------------------------")
                self.video_loader.dataset.train = True
        self.model.train()

        return accuracy_dict


    def prepare_task(self, task_dict, images_to_device = True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list
    # 此函數無用。
    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]


    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self, test_checkpoint_iter):
        if test_checkpoint_iter:
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(test_checkpoint_iter)))
        else:
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    main()
