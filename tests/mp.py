from multiprocessing import Pool, Queue
import multiprocessing
import threading
import time

def test(x):
    x0,x1,x2 = x
    time.sleep(2)
    return x0+x1+x2, x0*x1*x2
    # if p==10000:
    #     return True
    # else:
    #     return False


class Dog():
    def __init__(self):
        pass

    def go(self,name):
        return "hello %s" % name

def tt(n):
    n = "rensike"
    return

if __name__ == "__main__":
    # result=Queue() #队列
    # pool = Pool()
    # def pool_th():
    #     for i  in range(50000000): ##这里需要创建执行的子进程非常多
    #         try:
    #             result.put(pool.apply_async(test, args=(i,)))
    #         except:
    #             break
    # def result_th():
    #     while 1:
    #         a=result.get().get() #获取子进程返回值
    #         if a:
    #             pool.terminate() #结束所有子进程
    #             break
    # '''
    # 利用多线程，同时运行Pool函数创建执行子进程，以及运行获取子进程返回值函数。
    # '''
    # t1=threading.Thread(target=pool_th)
    # t2=threading.Thread(target=result_th)
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()
    # pool.join()

    # p = multiprocessing.Pool(1)
    # rslt = p.map(test, ((3,6,9),))
    # print(rslt[0])

    from multiprocessing import Pool
    # import numpy as np
    #
    # def foo(a):
    #     return "hello {}".format(a)
    #
    # p = Pool()
    # result = p.apply_async(foo, args=(np.random.random((2,3)),))
    # # foo和a分别是你的方法和参数，这行可以写多个，执行多个进程，返回不同结果
    # p.close()
    # p.join()
    #
    # r = result.get()
    #
    # print(r)

    # p = Pool()
    # d = Dog()
    # result = p.apply_async(d.go,args=("sikeppp",))
    #
    # p.close()
    # p.join()
    #
    # r = result.get()
    # print(r)
