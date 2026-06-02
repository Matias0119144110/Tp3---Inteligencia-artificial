import numpy as np

# ---------------------------------------------------------
# PROTOTIPO DE RED DE HOPFIELD PARA IDENTIFICAR UN ARO 10x10
# ---------------------------------------------------------

# Función para mostrar la matriz de forma más visual
def mostrar_matriz(matriz, titulo):
    print("\n" + titulo)
    for fila in matriz:
        linea = ""
        for valor in fila:
            if valor == 1:
                linea += "■ "
            else:
                linea += ". "
        print(linea)


# Función de activación signo
def signo(x):
    return np.where(x >= 0, 1, -1)


# ---------------------------------------------------------
# 1) Patrón original: aro simple en una matriz de 10x10
# ---------------------------------------------------------

aro_original = np.array([
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
    [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
    [-1,  1, -1, -1, -1, -1, -1, -1,  1, -1],
    [ 1, -1, -1, -1, -1, -1, -1, -1, -1,  1],
    [ 1, -1, -1, -1, -1, -1, -1, -1, -1,  1],
    [-1,  1, -1, -1, -1, -1, -1, -1,  1, -1],
    [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
    [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1]
])

# ---------------------------------------------------------
# 2) Entrenamiento de Hopfield con regla de Hebb
# ---------------------------------------------------------

# Convertimos la matriz 10x10 en un vector de 100 elementos
patron = aro_original.flatten()

# Cantidad de neuronas
n = patron.size

# Matriz de pesos usando Hebb
W = np.outer(patron, patron)

# Se anula la diagonal para evitar que una neurona se conecte consigo misma
np.fill_diagonal(W, 0)

# Normalización opcional para mantener valores controlados
W = W / n

# ---------------------------------------------------------
# 3) Generación de una imagen con ruido
# ---------------------------------------------------------

aro_ruidoso = aro_original.copy()

# Se alteran algunos píxeles para simular ruido o fallas en la imagen
pixeles_con_ruido = [
    (0, 0), (1, 4), (2, 7), (3, 3), (4, 9),
    (5, 1), (6, 8), (7, 2), (8, 5), (9, 9)
]

for fila, columna in pixeles_con_ruido:
    aro_ruidoso[fila, columna] *= -1

# ---------------------------------------------------------
# 4) Recuperación del patrón
# ---------------------------------------------------------

estado = aro_ruidoso.flatten()

# Iteraciones de recuperación
for iteracion in range(10):
    nuevo_estado = signo(W @ estado)

    # Si no hay cambios, la red llegó a un estado estable
    if np.array_equal(nuevo_estado, estado):
        print("\nLa red se estabilizó en la iteración:", iteracion + 1)
        break

    estado = nuevo_estado

aro_recuperado = estado.reshape(10, 10)

# ---------------------------------------------------------
# 5) Cálculo aproximado del centro del aro recuperado
# ---------------------------------------------------------

posiciones_activas = np.argwhere(aro_recuperado == 1)

centro_y = np.mean(posiciones_activas[:, 0])
centro_x = np.mean(posiciones_activas[:, 1])

# ---------------------------------------------------------
# 6) Salida por pantalla
# ---------------------------------------------------------

mostrar_matriz(aro_original, "PATRÓN ORIGINAL DEL ARO")
mostrar_matriz(aro_ruidoso, "PATRÓN CON RUIDO")
mostrar_matriz(aro_recuperado, "PATRÓN RECUPERADO POR HOPFIELD")

print("\nCentro aproximado del aro recuperado:")
print("Coordenada X:", round(centro_x, 2))
print("Coordenada Y:", round(centro_y, 2))