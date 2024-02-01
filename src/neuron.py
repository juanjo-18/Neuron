import math
class Neuron:
    def __init__(self, weights, bias, func):
        """
        Inicializa la neurona con los pesos, sesgo y función de activación.

        Parámetros:
        - weights: Lista de pesos para cada entrada.
        - bias: Sesgo de la neurona.
        - func: Función de activación ("relu", "sigmoid" o "tanh").
        """
        self.weights = weights
        self.bias = bias
        self.activation_function = func
        self.activation_functions = {
            "ReLu": self.__relu,
            "Sigmoide": self.__sigmoid,
            "TangenteHiperbólica": self.__tanh
        }

    def run(self, input_data):
        """
        Ejecuta la neurona con los datos de entrada y devuelve el resultado después de aplicar la función de activación.

        Parámetros:
        - input_data: Lista de datos de entrada.

        Retorna:
        - Resultado después de aplicar la función de activación.
        """
        # Verificar si la longitud de los pesos coincide con la longitud de los datos de entrada
        if len(self.weights) != len(input_data):
            raise ValueError("La cantidad no es la misma.")

        # Calcular la suma ponderada de los productos de los pesos y los datos de entrada
        weighted_sum = sum(w * x for w, x in zip(self.weights, input_data))

        # Agregar el sesgo
        weighted_sum += self.bias

        # Utilizar las funcion de activacion
        if self.activation_function in self.activation_functions:
            activacion_func = self.activation_functions[self.activation_function]
            output = activacion_func(Neuron,weighted_sum)
            return output
        else:
            raise ValueError("Función de activación no reconocida.")

    @staticmethod
    def __relu(self,weighted_sum):
        """
        Implementación de la función de activación ReLU.

        Parámetros:
        - weighted_sum: Suma ponderada de los productos de los pesos y los datos de entrada.

        Retorna:
        - Resultado después de aplicar la función ReLU.
        """
        return max(0, weighted_sum)

    @staticmethod
    def __sigmoid(self, weighted_sum):
        """
        Implementación de la función de activación sigmoide.

        Parámetros:
        - weighted_sum: Suma ponderada de los productos de los pesos y los datos de entrada.

        Retorna:
        - Resultado después de aplicar la función sigmoide.
        """
        return 1 / (1 + math.exp(-weighted_sum))

    @staticmethod
    def __tanh(self, weighted_sum):
        """
        Implementación de la función de activación tangente hiperbólica.

        Parámetros:
        - weighted_sum: Suma ponderada de los productos de los pesos y los datos de entrada.

        Retorna:
        - Resultado después de aplicar la función tangente hiperbólica.
        """
        return math.tanh(weighted_sum)


    def changeBias(self, new_bias):
        """
        Cambia el sesgo de la neurona.

        Parámetros:
        - new_bias: Nuevo valor para el sesgo.
        """
        self.bias = new_bias

    def changeWeight(self, new_weight):
        """
        Cambia los pesos de la neurona.

        Parámetros:
        - new_weight: Nuevos valores para los pesos.
        """
        self.weights = new_weight
