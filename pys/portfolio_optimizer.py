import numpy as np
from scipy.optimize import minimize

class PortfolioOptimizer:
    def init(self):
        pass

    @staticmethod
    def markowitz_optimization(expected_returns, cov_matrix, target_return=None):
        """
        Оптимизация портфеля по модели Марковица.
        
        Параметры:
          expected_returns - numpy array с ожидаемыми доходностями активов
          cov_matrix - numpy array с ковариационной матрицей активов
          target_return - целевая доходность портфеля (если указана)
          
        Возвращает:
          Оптимальные веса активов (numpy array)
        """
        n = len(expected_returns)
        initial_weights = np.ones(n) / n

        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return})
        bounds = [(0.0, 1.0)] * n

        result = minimize(portfolio_variance, initial_weights, bounds=bounds, constraints=constraints)
        if result.success:
            return result.x
        else:
            raise ValueError("Оптимизация по модели Марковица не удалась!")

    @staticmethod
    def black_litterman_allocation(market_weights, cov_matrix, tau, P, Q, omega):
        """
        Применяет упрощенную модель Блека–Литермана для корректировки весов портфеля.
        
        Параметры:
          market_weights - numpy array с рыночными весами активов
          cov_matrix - numpy array с ковариационной матрицей активов
          tau - скалярный параметр (например, 0.05)
          P - numpy array (матрица представлений)
          Q - numpy array (вектор представленных ожиданий доходностей)
          omega - numpy array или диагональная матрица, отражающая неопределенность представлений
          
        Возвращает:
          Скорректированные веса активов (numpy array)
        """
        pi = tau * cov_matrix @ market_weights

        middle_inv = np.linalg.inv(P @ (tau * cov_matrix) @ P.T + omega)
        adjusted_return = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + P.T @ middle_inv @ P) @ (
            np.linalg.inv(tau * cov_matrix) @ pi + P.T @ middle_inv @ Q
        )
        target_return = adjusted_return.mean()

        weights = PortfolioOptimizer.markowitz_optimization(adjusted_return, cov_matrix, target_return)
        return weights