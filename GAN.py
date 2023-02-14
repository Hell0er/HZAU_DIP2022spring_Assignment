import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns


sns.set(color_codes = True)

seed = 42
np.random.seed(seed)    #不同机器生成的随机数相同
tf.set_random_seed(seed)

class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5 
    
    def sample(self,N):
        samples=np.random.normal(self.mu,self.sigma,N)
        samples.sort()
        return samples 


#随机生成一些输入（噪点）
class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    #随机高斯初始化
    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
             np.random.random(N) * 0.01



def linear(input, output_dim, scope = None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(name='w', shape=[input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable(name='b', shape=[output_dim], initializer=const)
        return tf.matmul(input, w) + b
        
def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1

#return 预测结果：为真为假的可能性分别为多少
def discriminator(input,h_dim):
    h0=tf.tanh(linear(input, h_dim * 2,'d0'))
    h1=tf.tanh(linear(h0, h_dim * 2,'d1'))
    h2=tf.tanh(linear(h1, h_dim * 2,scope='d2'))

    h3=tf.sigmoid(linear(h2,1,scope='d3'))
    return h3

#迭代器
def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.95
    num_decay_strps = 150
    batch= tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_strps,
        decay,
        staircase=True
    )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    
    return optimizer



def main(args):
    model = GAN (
        DataDistribution(),
        GeneratorDistribution(range=8),
        args.num_steps,
        args.batch_size,
        args.log_every,
    )
    model.train()


#定义了一些默认值比如 num-steps:迭代次数
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--num-steps',type=int,default=1200,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size',type=int,default=12,
                        help='the batch size')
    parser.add_argument('--log-every',type=int,default=10,
                        help='print loss after this many steps')
    return parser.parse_args()



class GAN(object):

    def __init__(self, data, gen, num_steps, batch_size, log_every):
        self.data = data
        self.gen = gen 
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.mlp_hidden_size = 4

        self.learning_rate = 0.03  #学习率
        self._create_model()
    
    def _create_model(self):

        #先预训练一个模型
        with tf.variable_creator_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        #Gen网络生成数据
        with tf.variable_scope("Gen"):
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.G = generator(self.z, self.mlp_hidden_size)

        #Disa网络，接受G(x)和真实数据
        with tf.variable_scope('Disc') as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            
            #self.x真实数据
            #self.D1对真实函数判断结果
            self.D1 = discriminator(self.x, self.mlp_hidden_size)
            scope.reuse_variables()

            #self.G生成数据
            self.D2 = discriminator(self.G, self.mlp_hidden_size)

        #损失函数
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'D_pre')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)


    def _samples(self,session,num_points=10000,num_bins=100):
        xs=np.linspace(-self.gen.range,self.gen.range,num_points)
        bins=np.linspace(-self.gen.range,self.gen.range,num_bins)


        d=self.data.sample(num_points)
        pd, _ =np.histogram(d,bins=bins,density=True)


        zs=np.linspace(-self.gen.range,self.gen.range,num_points)
        g=np.zeros((num_points,1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size*i:self.batch_size*(i+1)]=session.run(self.G, {
                self.z:np.reshape(
                    zs[self.batch_size * i:self.batch_size*(i+1)],
                    (self.batch_size,1)
                )
            })
        pg, _ =np.histogram(g,bins=bins,density=True)

        return pd,pg
    
    def _plot_distribution(self, session):
        pd, pg = self._samples(session)
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label="real data")
        plt.plot(p_x, pg, lable="generated data")
        plt.title("ID Generative Adversarial Network")
        plt.xlabel("Data values")
        plt.ylabel("Probability density")
        plt.legend()
        plt.show()
        
    def train(self):
        with tf.Session() as session:
            tf.global_variables_initializer().run()

            num_pretrain_steps = 1000
            for step in range(num_pretrain_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })
            self.weightsD = session.run(self.d_pre_params)

            #权重参数拷贝
            for i, v in enumerate(self.d_params):
                    session.run(v.assign(self.weightsD[i]))

            for step in range(self.num_steps):

            #更新disc模型
                x = self.data.sample(self.batch_size)
                z = self.gen.sample(self.batch_size)
                loss_d, _ = session.run([self.batch_size, self.opt_d], {
                        self.x: np.reshape(x, (self.batch_size, 1)),
                        self.z: np.reshape(z, (self.batch_size, 1))
                    })

                #更新gen模型
                z = self.gen.sample(self.batch_size)  #随机初始化
                loss_g, _ = session.run([self.loss_g, self.opt_g], {     #不断迭代优化
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                if step % self.log_every == 0:
                    print('{}: {} \t{}'.format(step, loss_d, loss_g))
                if step % 100 == 0 or step == 0 or step == self.num_steps -1:
                    self._plot_distribution(session)


if __name__ == '__main__':
    main(parse_args())

