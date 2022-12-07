import random
from imp import reload
from gensim import corpora
from jieba import xrange
from keras.datasets import imdb
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) ##去除numpy警告




(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


##将train_data转化为英文
def Turn_to_English(data):
    English = []
    for j in enumerate(data):
       English.append(' '.join([reverse_word_index.get(i - 3, '?') for i in j[1]]))
    return English


# 词袋模型
def Bag_of_words_model(data):
    word_vocab = []
    for i in word_index:
        word_vocab.append(i)
    del word_vocab[10000:len(word_vocab)]            #制作词表
    count = CountVectorizer()
    count.fit(word_vocab)
    Tool1_fit = count.transform(data)
    return Tool1_fit.toarray()


#感知分类器
def perceptron():
   X_train = Bag_of_words_model(Turn_to_English(train_data))
   Kuolie = []
   Kuolie1 = np.array([])
   for i in range(25000):
       Kuolie.append(0)
   Kuolie1=np.append(Kuolie1,Kuolie)
   for i in range(247):
       X_train = np.c_[X_train, Kuolie1.T]           ##补足10000位
   Demo = Perceptron(fit_intercept = True, max_iter = 1000, tol = 0.001, shuffle = False, eta0 = 0.1)
   #fit_intercept是否对参数截距项b进行估计，若为false数据为中心化
   #max_iter最大迭代次数，tol停止迭代标准，shuffle每次训练后是否打乱数据，eta0学习率，决定每次参数变化的幅度
   Demo.fit(X_train,train_labels)
   #print(len(Demo.intercept_))
   return Demo


pc = 0.02  # pc为变异的概率
t = 10  # 遗传算法迭代的次数
n = 50 # 种群的个体数,要求大于20以保证具有随机性
#遗传算法
def GA():
    X_test = Bag_of_words_model(Turn_to_English(test_data))
    Kuolie = []
    Kuolie1 = np.array([])
    for i in range(25000):
        Kuolie.append(0)
    Kuolie1 = np.append(Kuolie1, Kuolie)
    for i in range(247):
        X_test = np.c_[X_test, Kuolie1.T]            #补足10000位

    First_Population = []                #随机生成n个个体，作为初始种群
    for i in range(n):
        number = random.randint(0, 25000)
        First_Population.append(number)

    # 第一代种群
    population=np.zeros((n,10000))
    for i in range(n):
        population[i] = X_test[First_Population[i]]


    # 遗传算法的迭代次数为t
    fitness_num=np.zeros(n*t)   #存储所有个体的适应值
    fitness_change = np.zeros(t)
    for i in range(t):
        fitness = np.zeros(n)  # fitness为每一个个体的适应度值
        for j in range(n):
            fitness[j] = Jd(population[j])  # 计算每一个体的适应度值
        #是否满足停止迭代条件
        for m in range(n):
            fitness_num[i*n+m] = fitness[m]   #存储每代个体的适应值
        result = np.sum(fitness == fitness[0]) == len(fitness)
        if result:
            break

        population = selection(population, fitness)  # 通过概率选择产生新一代的种群
        population = crossover(population)  # 通过交叉产生新的个体
        population = mutation(population)  # 通过变异产生新个体
        fitness_change[i] = max(fitness)  # 找出每一代的适应度最大的染色体的适应度值
        print("第",i,"次的最大适应度为：")
        print(fitness_change[i])
        print("第",i,"次的最优染色体为：")
        print(population[fitness.argmax()])  #np.argmax  取出其中最大值的所对应的索引
    # 画图
    plt.rcParams['axes.unicode_minus'] = False
    list_x = []
    for x in range(n*t):
        list_x.append(x)
    plt.xlabel('dimension')
    plt.ylabel('fitness')
    plt.axis([0, n*t, -1000, 1000])
    plt.scatter(list_x, fitness_num, color='r')  # 画图
    plt.show()
    best_fitness = max(fitness_change)
    best_people = population[fitness.argmax()]

    return best_people, best_fitness, fitness_change, population

# 轮盘赌选择
#个体排序顺序会影响“轮盘赌”法；当适应性值相差不大的情况，该方法与随机选没什么区别，
# 此时难于控制进化方向和速度；该方法是属于有放回的选择，不适合不放回的情况。
def selection(population, fitness):
    fitness_sum = np.zeros(n)
    for i in range(n):
        if i == 0:
            fitness_sum[i] = fitness[i]
        else:
            fitness_sum[i] = fitness[i] + fitness_sum[i - 1]     #计算累加适应值
    for i in range(n):
        fitness_sum[i] = fitness_sum[i] / sum(fitness)    #计算累计概率
    # 选择新的种群
    population_new = np.zeros((n, 10000))
    for i in range(n):
        rand = np.random.uniform(-1, 1)      #随机生成一个实数
        for j in range(n):
            if j == 0:
                if rand <= fitness_sum[j]:
                    population_new[i] = population[j]
            else:
                if fitness_sum[j - 1] < rand and rand <= fitness_sum[j]:
                    population_new[i] = population[j]
    return population_new

# 交叉操作
def crossover(population):
    m=int(n/2)
    father = population[0:m, :]
    mother = population[m:, :]
    np.random.shuffle(father)  # 将父代个体按行打乱以随机配对
    np.random.shuffle(mother)
    for i in range(m):
        father_1 = []                                        #用father_1,mother_1临时代替个体
        mother_1 = []
        father_1 = father[i]
        mother_1 = mother[i]
        one_zero = []                                       #用one_zero ,zero_one记录个体中不一样的下标
        zero_one = []
        for j in range(5000):                               #选择一半进行交叉
            if father_1[j] == 1 and mother_1[j] == 0:
                one_zero.append(j)
            if father_1[j] == 0 and mother_1[j] == 1:
                zero_one.append(j)
        length1 = len(one_zero)
        length2 = len(zero_one)
        length = min(length1, length2)
        for k in range(length):  # 进行交叉操作
            p = one_zero[k]
            q = zero_one[k]
            father_1[p] = 0
            mother_1[p] = 10
            father_1[q] = 1
            mother_1[q] = 0
        father[i] = father_1  # 将交叉后的个体替换原来的个体
        mother[i] = mother_1
    population = np.append(father, mother, axis=0)
    return population

# 变异操作
def mutation(population):
    for i in range(n):
        c = np.random.uniform(0, 1)
        if c <= pc:
            mutation_s = population[i]
            zero = []  # zero存的是变异个体中第几个数为0
            one = []  # one存的是变异个体中第几个数为1
            for j in range(10000):
                if mutation_s[j] == 0:
                    zero.append(j)
                else:
                    one.append(j)
            a = np.random.randint(0, len(zero))  # e是随机选择由0变为1的位置
            b = np.random.randint(0, len(one))  # f是随机选择由1变为0的位置
            e = zero[a]
            f = one[b]
            mutation_s[e] = 1
            mutation_s[f] = 0
            population[i] = mutation_s
    return population


# 个体适应度函数 Jd(x)
def Jd(x):
    Adaptation_value = perceptron()  # train计算的权重，用于个体适应度
    dotx=np.dot(Adaptation_value.coef_, x)             #coef_	array 二维数组	输出训练后的模型参数w的数组，不包含截距项b。
                                                     # 当为二分类时，该数组shape=(1,n)，n为特征数量。当为多分类时shape=（k, n)
    dotx = dotx+Adaptation_value.intercept_[0]

    print("适应度:")
    print(dotx)
    return dotx


if __name__ == '__main__':

    # best_d = np.zeros(d)          # judge存的是每一个维数的最优适应度

    # fitness_change是遗传算法在迭代过程中适应度变化
    # best是每一维数迭代到最后的最优的适应度，用于比较
    best_people, best_fitness, fitness_change, best_population = GA()
    print("最大适应度为：")
    print(best_fitness)
    print("选出的最优染色体为：")
    print(best_people)






