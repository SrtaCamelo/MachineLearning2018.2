import numpy as np
from SwarmPackagePy.intelligence import sw


class pso(sw):
    def __init__(self, n, function, lb, ub, dimension, iteration, init=None, w=0.5, c1=1,
                 c2=1):

        super(pso, self).__init__()

        if init is None:
            self.__agents = np.random.uniform(lb, ub, (n, dimension))
        else:
            self.__agents = np.array(init)

        velocity = np.zeros((n, dimension))
        self._points(self.__agents)

        Pbest = self.__agents[np.array([function(x)
                                        for x in self.__agents]).argmin()]
        Gbest = Pbest

        for t in range(iteration):

            r1 = np.random.random((n, dimension))
            r2 = np.random.random((n, dimension))
            velocity = w * velocity + c1 * r1 * (
                Pbest - self.__agents) + c2 * r2 * (
                Gbest - self.__agents)
            self.__agents += velocity
            self.__agents = np.clip(self.__agents, lb, ub)
            self._points(self.__agents)

            Pbest = self.__agents[
                np.array([function(x) for x in self.__agents]).argmin()]
            if function(Pbest) < function(Gbest):
                Gbest = Pbest

        self._set_Gbest(Gbest)