class Evaluation_func:
    def __init__(self, x_threshold):
        self.x_threshold = x_threshold
        self.f_values = []

    def calculate_f(self, yip, yit):
        ratio = abs(yip - yit) / yit
        f_value = ratio if ratio > self.x_threshold else 0
        self.f_values.append(f_value)
        return f_value

    def calculate_y_hat(self):
        sum_fi = sum(self.f_values)
        N = len(self.f_values)
        avg_fi = sum_fi / N if N > 0 else float('inf')
        y_hat = 1 / avg_fi if avg_fi != 0 else float('inf')
        return y_hat
