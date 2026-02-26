# -*- coding: utf-8 -*-
"""
Logistic Regression
Classificação binária (0/1)
Função de ativação: Sigmoid
Função custo: Cross-Entropy (Log Loss)
"""

import numpy as np
from dataset import Dataset
import matplotlib.pyplot as plt

class LogisticRegression:
    
    def __init__(self, dataset, standardize=False):
        """
        dataset → objeto Dataset com X e Y
        standardize → se True, normaliza as features
        """

        # Se quisermos normalizar as features
        if standardize:
            dataset.standardize()
            # adiciona coluna de 1's (bias) + dados normalizados
            self.X = np.hstack((np.ones([dataset.nrows(),1]), dataset.Xst))
            self.standardized = True
        else:
            # adiciona coluna de 1's (bias) + dados originais
            self.X = np.hstack((np.ones([dataset.nrows(),1]), dataset.X))
            self.standardized = False

        # vetor de outputs (0 ou 1)
        self.y = dataset.Y

        # inicializa parâmetros θ com zeros
        self.theta = np.zeros(self.X.shape[1])

        # guardar dataset (para usar média e desvio se normalizado)
        self.data = dataset


    
    def probability(self, instance):
        """
        Calcula P(y=1 | x)
        """

        # criar vetor x com bias
        x = np.empty([self.X.shape[1]])
        x[0] = 1  # bias
        x[1:] = np.array(instance[:self.X.shape[1]-1])

        # se o modelo foi treinado com normalização,
        # também temos de normalizar este exemplo
        if self.standardized:
            if np.all(self.data.sigma != 0):
                x[1:] = (x[1:] - self.data.mu) / self.data.sigma
            else:
                x[1:] = (x[1:] - self.data.mu)

        # calcular θᵀx
        z = np.dot(self.theta, x)

        # aplicar sigmoid → probabilidade
        return sigmoid(z)


    def predict(self, instance):
        """
        Se probabilidade >= 0.5 → classe 1
        Caso contrário → classe 0
        """
        p = self.probability(instance)

        if p >= 0.5:
            return 1
        else:
            return 0

    def costFunction(self, theta=None):
        """
        Implementa:
        J(θ) = -(1/m) ∑ [ y log(h) + (1-y) log(1-h) ]
        """

        if theta is None:
            theta = self.theta

        m = self.X.shape[0]

        # calcular h = sigmoid(Xθ)
        p = sigmoid(np.dot(self.X, theta))

        # aplicar fórmula da log-loss
        cost = (-self.y * np.log(p) -
                (1 - self.y) * np.log(1 - p))

        # média
        return np.sum(cost) / m


    def gradientDescent(self, alpha=0.01, iters=10000):
        """
        Atualiza θ usando:
        θ := θ - α/m * Xᵀ(h - y)
        """

        m = self.X.shape[0]
        n = self.X.shape[1]

        # reinicializa θ
        self.theta = np.zeros(n)

        for its in range(iters):

            # opcional: mostrar custo de 1000 em 1000 iterações
            if its % 1000 == 0:
                print(self.costFunction())

            # h = sigmoid(Xθ)
            h = sigmoid(self.X.dot(self.theta))

            # gradiente = Xᵀ(h - y)
            delta = self.X.T.dot(h - self.y)

            # atualização dos parâmetros
            self.theta -= (alpha / m) * delta


    def buildModel(self):
        """
        Usa método de otimização do scipy
        em vez de gradient descent manual
        """
        self.optim_model()


    def optim_model(self):
        from scipy import optimize

        n = self.X.shape[1]
        initial_theta = np.zeros(n)

        # minimiza a função custo automaticamente
        self.theta, _, _, _, _ = optimize.fmin(
            lambda theta: self.costFunction(theta),
            initial_theta,
            maxiter=500,
            full_output=True
        )

    def printCoefs(self):
        print(self.theta)

    def plotModel(self):
        """
        Só funciona para 2 variáveis.
        Desenha:
        - Pontos classe 0
        - Pontos classe 1
        - Linha de decisão θᵀx = 0
        """

        from numpy import r_

        pos = (self.y == 1).nonzero()[:1]
        neg = (self.y == 0).nonzero()[:1]

        # desenhar pontos
        plt.plot(self.X[pos, 1].T, self.X[pos, 2].T,
                 'k+', markeredgewidth=2, markersize=7)

        plt.plot(self.X[neg, 1].T, self.X[neg, 2].T,
                 'ko', markerfacecolor='r', markersize=7)

        # desenhar linha θ0 + θ1x1 + θ2x2 = 0
        if self.X.shape[1] <= 3:
            plot_x = r_[self.X[:,2].min(), self.X[:,2].max()]
            plot_y = (-1./self.theta[2]) * \
                     (self.theta[1]*plot_x + self.theta[0])

            plt.plot(plot_x, plot_y)
            plt.legend(['class 1', 'class 0', 'Decision Boundary'])

        plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def test():

    ds = Dataset("log-ex1.data")

    logmodel = LogisticRegression(ds)

    print("Initial cost:", logmodel.costFunction())
    # esperado ≈ 0.693 (modelo aleatório)

    # treinar com gradient descent
    logmodel.gradientDescent(0.002, 400000)

    logmodel.plotModel()

    print("Final cost:", logmodel.costFunction())

    ex = np.array([45, 65])

    print("Probabilidade:", logmodel.probability(ex))
    print("Classe prevista:", logmodel.predict(ex))


if __name__ == '__main__':
    test()
