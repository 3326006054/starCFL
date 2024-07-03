from copy import deepcopy
from random import random
import argparse
import time
from torch import cosine_similarity
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.utils import SerializationTool, Aggregators, Logger
from tqdm import tqdm

from models.cnn import CNN_CIFAR10, CNN_MNIST, AlexNet_CIFAR10, CNN_FEMNIST
from models.linear import SimpleLinear
from functional import *
from trainer import FEMNISTTrainer, ShiftedMNISTTrainer, RotatedMNISTTrainer, LabelSkewMNISTTrainer, \
    FeatureSkewHybridMNISTTrainer
from datasets import BaseDataset, RotatedMNIST, RotatedCIFAR10Partitioner

import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from fedlab.core.client.serial_trainer import SerialTrainer
from newstuff import CIFAR10Partitioner, MNISTPartitioner  # 宋楠的数据集


class Client(SerialTrainer):
    def __init__(self, model, args, client_id, server, cuda=False, logger=None):
        super().__init__(model, 0, cuda, logger)
        self.args = args
        self.client_id = client_id
        self.cluster = None
        self.server = server  # 每一个客户端都对应一个服务端
        self.data_loader = self.server.dataset.get_data_loader(client_id, self.args.obbs)  # 初始化data_loader
        self.features = None  # 特征在分类的时候初始化
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)
        self.clustered = False  # 是否已经被聚类

    def train_alone_prox(self):
        criterion = torch.nn.CrossEntropyLoss()
        """ 下面一条语句我注释掉了， 一次客户端单独的训练不需要直接分发cluster的模型， 这个有待商榷"""
        # SerializationTool.deserialize_model(self._model, self.cluster.model_parameters)

        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, self.server.model_parameters)

        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in self.data_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                l1 = criterion(preds, label)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * self.args.mu * l2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.model_parameters

    def extract_features(self):
        if self.features is not None:
            return self.features
        else:
            features = []
            # 获取模型特征
            with torch.no_grad():
                for inputs, _ in self.data_loader:
                    outputs = self.server.resnet18(inputs)
                    features.append(outputs.numpy())
            features = np.concatenate(features, axis=0)
            self.features = features
            return features  # 是否需要返回特征？

    def __lt__(self, other):
        return self.client_id < other.client_id





class Cluster(ModelMaintainer):
    def __init__(self, cluster_id, server, args):
        super().__init__(server._model, server.cuda)
        self.args = args
        self.cluster_id = cluster_id
        self.server = server
        self._model = deepcopy(server._model)
        self.clients = []
        SerializationTool.deserialize_model(self._model, self.server.model_parameters)

    def aggregators(self):
        updates = [client.model_parameters * (1 - self.args.a) + self.server.model_parameters * self.args.a for client
                   in self.clients]
        SerializationTool.deserialize_model(self._model, Aggregators.fedavg_aggregate(updates))

    def aggregators_avg(self):
        updates = [client.model_parameters for client in self.clients]
        SerializationTool.deserialize_model(self._model, Aggregators.fedavg_aggregate(updates))

    def distribute_model(self):
        """
            将聚类的模型分发给每一个客户端
        """
        for client in self.clients:
            SerializationTool.deserialize_model(client._model, self.model_parameters)

    def train_in_cluster(self):
        """
            在新建的聚类中训练参加训练的客户端
        """
        for t in range(self.args.daisy_rounds):
            for client in tqdm(self.clients):
                client.train_alone_prox()
            # 每隔d随机分发模型
            if t % self.args.d == self.args.d - 1:
                client_models = [client.model_parameters for client in self.clients]
                random.shuffle(client_models)
                for i in range(len(self.clients)):
                    SerializationTool.deserialize_model(self.clients[i]._model, client_models[i])
            # 每隔b轮聚合模型
            if t % self.args.b == self.args.b - 1:
                updates = [client.model_parameters for client in self.clients]
                SerializationTool.deserialize_model(self._model, Aggregators.fedavg_aggregate(updates))
                self.distribute_model()
                """
                test_loader = self.server.dataset.get_data_loader(
                    int(int(self.clients[0].client_id) * self.args.k / self.args.n), type='test')
                loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), test_loader)
                self.args.exp_logger.info(
                    "Round {}, Cluster {} - Cluster Test loss: {:.4f}, acc: {:.4f}".format(t, self.cluster_id, loss, acc))
                """

    def append_client(self, client):
        """
            将客户端添加到聚类中
        """
        self.clients.append(client)
        client.cluster = self


class Server(ModelMaintainer):
    def __init__(self, args, model, dataset, cuda=False):
        super().__init__(model, cuda)
        self.args = args
        self.clients = []  # 客户端的集合
        self.clusters = []  # 聚类的集合
        self.gmodel = deepcopy(self.model_parameters)  # 这个感觉不需要也可以， 全局模型可以用self.model_parameters获得
        self.dataset = dataset  # 数据集
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Identity()
        resnet18 = resnet18.eval()
        self.resnet18 = resnet18
        self.clients_this_round = None  # 选取的客户端
        self.cid_this_round = None  # 选取的客户端的id

    def sample_clients(self):
        """
            随机选取未聚类的的客户端客户端
            参数: None
            返回: 选取的客户端类的引用
        """
        unclustered_clients = [client for client in self.clients if client.clustered is False]
        self.clients_this_round = sorted(random.sample(unclustered_clients, self.args.num_per_round))  # 选取客户端的类的应用
        self.cid_this_round = [client.client_id for client in self.clients_this_round]  # 选取客户端的id
        return self.clients_this_round

    @staticmethod
    def compute_model_cosine(model1_params, model2_params):
        """
            计算两个模型的余弦相似度
        """
        model1_params = model1_params.flatten()
        model2_params = model2_params.flatten()
        imilarity = cosine_similarity(model1_params.unsqueeze(0), model2_params.unsqueeze(0))[0][0]
        return imilarity

    def compute_cosine_similarity(self, features_list):
        num_clients = len(features_list)
        similarity_matrix = np.zeros((num_clients, num_clients))

        for i in range(num_clients):
            for j in range(i, num_clients):
                sim = cosine_similarity([features_list[i].flatten()], [features_list[j].flatten()])[0][0]
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix

    def find_optimal_clusters(self, data, max_k):
        iters = range(1, max_k + 1)
        sse = []

        for k in iters:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            sse.append(kmeans.inertia_)
        # 寻找手肘点
        x = np.arange(len(sse))
        y = np.array(sse)
        # 计算二阶差分来找到拐点
        diff = np.diff(y, 2)
        best_k = np.argmin(diff) + 2  # +2 是因为差分数组的索引偏移和 range 从1开始

        plt.figure()
        plt.plot(iters, sse, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.title('SSE for Different Number of Clusters')
        plt.show()

        print(f'shou number of clusters: {best_k}')

        return best_k

    def find_cluster(self, cluster_id):
        """
            根据cluster_id找到对应的cluster
        """
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None

    def getCluster(self):
        # 后面再加参数
        client_features = [client.extract_features() for client in self.clients_this_round]
        cosine_similarity_matrix = self.compute_cosine_similarity(client_features)
        optimal_k = self.find_optimal_clusters(cosine_similarity_matrix, max_k=8)
        # 使用 KMeans 进行聚类
        kmeans = KMeans(n_clusters=optimal_k)
        clusters = kmeans.fit_predict(cosine_similarity_matrix)
        """ 将聚类添加到中心服务器中 """
        for i in range(optimal_k):
            cluster = Cluster(i, self, self.args)
            self.clusters.append(cluster)
        """ 将客户端添加到聚类中 """
        for client, cluster_id in zip(self.clients_this_round, clusters):
            cluster = self.find_cluster(cluster_id)
            cluster.append_client(client)
            client.clustered = True
        return self.clients_this_round

    def aggregators(self):
        """
            聚合参加训练的聚类的模型
        """
        updates = [cluster.model_parameters for cluster in self.clusters]
        weights = []
        for cluster in self.clusters:
            count = 0
            for client in cluster.clients:
                if client in self.clients_this_round:
                    count += 1
            weights.append(count)
        SerializationTool.deserialize_model(self._model, Aggregators.fedavg_aggregate(updates, weights=weights))

    def aggregators_without_weight(self):
        """
            聚合参加训练的聚类的模型
        """
        updates = [cluster.model_parameters for cluster in self.clusters]
        SerializationTool.deserialize_model(self._model, Aggregators.fedavg_aggregate(updates))

    def distribute_model(self):
        for cluster in self.clusters:
            SerializationTool.deserialize_model(cluster._model, self.model_parameters)

    def main(self):
        global_cluster_num = 0
        for round in range(self.args.com_round):
            # 选取客户端
            self.sample_clients()
            self.args.exp_logger.info(
                "Starting round {}/{}, client id this round {}".format(round, self.args.com_round, self.cid_this_round))
            self.getCluster()
            for i in range(5):
                for cluster in self.clusters:
                    cluster.train_in_cluster()
                self.aggregators()
                self.distribute_model()
            self.clusters.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone training example")
    # server
    parser.add_argument("--com_round", type=int)  # 通信轮数
    parser.add_argument("--num_per_round", type=int)  # 每次选取的客户端数量

    # trainer
    parser.add_argument("--batch_size", type=int)  # 批量大小
    parser.add_argument("--lr", type=float, default=0.1)  # 学习率
    parser.add_argument("--epochs", type=int, default=5)  # 训练轮数
    parser.add_argument("--mu", type=float, default=0.05)  # 正则化参数
    parser.add_argument("--n", type=int, default=200)  # 客户端数量

    # cluster
    parser.add_argument("--obbs", type=int, default=None)  # 每个client的样本数
    parser.add_argument("--train", type=int, default=1)  # 训练模式 # set this to 0, only clustering no learning.
    parser.add_argument("--seed", type=int)  # 随机种子

    parser.add_argument("--dataset", type=str, default="cifar")  # minist, cifar# mnist, cifar
    # datset
    parser.add_argument("--process", type=int, default=0)  # 是否预处理数据

    # feddc
    parser.add_argument("--daisy_rounds", type=int, default=15)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--b", type=int, default=5)
    parser.add_argument("--a", type=float, default=0.3)  # 个性化聚合参数

    # anchor
    args = parser.parse_args()  # 解析参数

    # args.seed = 0 # [0, 42, 1998, 4421380, 789190]
    setup_seed(args.seed)  # 设置随机种子

    args.root = "./datasets/{}/".format(args.dataset)  # 数据集路径
    args.save_dir = "./datasets/rotated_{}_{}_{}/".format(args.dataset, args.n, args.seed)  # 保存路径

    if args.dataset == "mnist":
        args.k = 4
        model = SimpleLinear()
        trainer = RotatedMNISTTrainer(deepcopy(model), args, cuda=False)

    if args.dataset == "cifar":
        args.k = 2
        model = CNN_CIFAR10()

    if args.process:
        print("Preprocessing datasets...")
        if args.dataset == "mnist":
            dataset = MNISTPartitioner(args.root, args.save_dir)
            # dataset = RotatedMNISTPartitioner(args.root, args.save_dir)
            dataset.pre_process()
        if args.dataset == "cifar":
            dataset = CIFAR10Partitioner(args.root, args.save_dir)  # <--- 宋楠的数据集
            dataset.pre_process(shards=int(args.n / 2))
    else:
        print("Preprocessing datasets...")
        if args.dataset == "mnist":
            dataset = RotatedMNIST(args.root, args.save_dir)
            # dataset = RotatedMNISTPartitioner(args.root, args.save_dir)
            dataset.pre_process(shards=int(args.n / 4))
        if args.dataset == "cifar":
            dataset = RotatedCIFAR10Partitioner(args.root, args.save_dir)
            dataset.pre_process(shards=int(args.n / 2))

    args.time_stamp = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    dir = "./logs/runs-rotated-{}-n{}-p{}-seed{}-lambda{}-time-{}".format(args.dataset, args.n, args.num_per_round,
                                                                          args.seed, args.mu, args.time_stamp)
    os.mkdir(dir)
    args.exp_logger = Logger("StoCFL", "{}/rotated_{}.log".format(dir, args.dataset))
    args.exp_logger.info(str(args))
    args.dir = dir

    Server = Server(args, deepcopy(model), dataset, False)
    for i in range(int(args.n)):
        client = Client(deepcopy(model), args, i, Server, False)
        Server.clients.append(client)
    Server.main()
