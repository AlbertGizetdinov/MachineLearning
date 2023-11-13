import sys
import matplotlib.pyplot as plt
import random
import numpy as np
import operator
from sklearn.cluster import KMeans

# Функция generatePoints генерирует заданное количество случайных точек на плоскости
def generate_points(amount_of_points=200):
    points = []
    for i in range(amount_of_points):
        points.append([random.randint(0, 100), random.randint(0, 100)])
    return points

# Функция display_points отображает точки на плоскости с помощью библиотеки matplotlib, а также отображает центроиды кластеров
def display_points(points, centroids, clusters=None):
    colors = ['b', 'g', 'y', 'm', 'c']
    for number_point in range(len(points)):
        color = colors[clusters[number_point]] if clusters else 'k'
        plt.scatter(points[number_point][0], points[number_point][1], color=color)
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], color='r')
    plt.show()

# Функция get_distance вычисляет Евклидово расстояние между двумя точками на плоскости
def get_distance(point_first, point_second):
    return np.sqrt((point_first[0] - point_second[0]) ** 2 +
                   (point_first[1] - point_second[1]) ** 2)

# Функция place_centroids располагает центроиды равномерно по окружности, центр которой находится в центре масс точек,
# а ее радиус равен расстоянию до самой удаленной точки
def place_centroids(points, amount_of_centroids=4):
    """
    Метод возвращает массив центроидов, расположенных равномерно по окружности.
    """
    center, circle_radius = calculate_mass_center_and_radius(points)
    centroids = []
    for i in range(amount_of_centroids):
        centroids.append([circle_radius * np.cos(2 * np.pi * i / amount_of_centroids) + center[0],
                          circle_radius * np.sin(2 * np.pi * i / amount_of_centroids) + center[1]])
    return centroids

# Функция calculate_mass_center_and_radius вычисляет центр масс точек и радиус окружности, описывающей эти точки
def calculate_mass_center_and_radius(points):
    """
    Метод возвращает центр окружности, центр которой лежит на среднем значении х и у точек,
    и радиус этой окружности, равный расстоянию от центра окружности до самой удалённой точки.
    """
    circle_center = [0, 0]
    for point in points:
        circle_center[0] += point[0]
        circle_center[1] += point[1]
    circle_center[0] /= len(points)
    circle_center[1] /= len(points)

    circle_radius = 0
    for point in points:
        distance_to_point = get_distance(circle_center, point)
        if distance_to_point > circle_radius:
            circle_radius = distance_to_point
    return [circle_center, circle_radius]

# Функция assign_clusters_to_points определяет, к какому кластеру принадлежит каждая точка, и возвращает массив номеров кластеров для каждой точки
def assign_clusters_to_points(points, centroids):
    """
    Метод ищет ближайшие точки к центроидам и возвращает массив,
    индексы которого равны индексам точек, а значение - номеру центроида
    """
    clusters = []
    for point in points:
        min_distance_to_centroid = sys.maxsize
        index = -1
        for centroid_number in range(len(centroids)):
            distance_from_centroid_to_point = get_distance(point, centroids[centroid_number])
            if min_distance_to_centroid > distance_from_centroid_to_point:
                min_distance_to_centroid = distance_from_centroid_to_point
                index = centroid_number
        clusters.append(index)
    return clusters

# Функция move_centroids вычисляет новые координаты центроидов на основе текущих кластеров и точек
def move_centroids(points, clusters):
    """
    Метод возвращает новые координаты центроидов для заданных кластеров
    """
    def divide_by_points_amount(array):
        return list(map(lambda sum_of_points: round(sum_of_points / array[2], 1), array[:2]))

    count_of_clusters = len(set(clusters))
    sum_of_coordinates = []
    for number_of_cluster in range(count_of_clusters):
        sum_of_coordinates.append([0, 0, 0])
    for number_point in range(len(points)):
        sum_of_coordinates[clusters[number_point]][0] += points[number_point][0]
        sum_of_coordinates[clusters[number_point]][1] += points[number_point][1]
        sum_of_coordinates[clusters[number_point]][2] += 1
    return list(map(divide_by_points_amount, sum_of_coordinates))

# Функция get_max_difference_between_arrays вычисляет максимальную разницу между соответствующими элементами двух массивов
def get_max_difference_between_arrays(first_array, second_array):
    max_difference = 0
    for index in range(len(first_array)):
        for inner_index in range(len(first_array[index])):
            difference = operator.abs(first_array[index][inner_index] - second_array[index][inner_index])
            if difference > max_difference:
                max_difference = difference
    return max_difference


def get_sum_of_square_distances(points, centroids, clusters):
    """
    Метод считает сумму квадратов расстояний для заданных точек и центроидов
    """
    # Distances between each point and corresponding centroid
    distances = [0] * len(centroids)
    total_distance = 0

    for point_index in range(len(points)):
        distances[clusters[point_index]] = get_distance(
            points[point_index],
            centroids[clusters[point_index]]
        )
    for index in range(len(distances)):
        total_distance += distances[index]
    return total_distance


if __name__ == '__main__':
    points = generate_points()
    sums_of_square_distances = [0] * 8
    for centroids_amount in range(len(sums_of_square_distances)):
        centroids_amount += 2
        centroids = place_centroids(points, centroids_amount)
        clusters = []
        # Moving centroids till they are static
        while True:
            new_clusters = assign_clusters_to_points(points, centroids)
            if np.array_equiv(clusters, new_clusters):
                break
            clusters = new_clusters
            new_centroids = move_centroids(points, clusters)
            centroids = new_centroids
        # Looking for J(C) for each k
        sums_of_square_distances[centroids_amount - 2] = get_sum_of_square_distances(points, centroids, clusters)

    # Calculating D(k)
    fall_rates_measures = [0] * 8
    for k_index, sum_of_squares_of_distances in enumerate(sums_of_square_distances[1:-1], start=1):
        fall_rates_measures[k_index] = operator.abs(
            sum_of_squares_of_distances - sums_of_square_distances[k_index + 1]) / operator.abs(
            sums_of_square_distances[k_index - 1] - sum_of_squares_of_distances)

    # Looking for a minimal D(k) to find optimal amount of clusters k
    optimal_k = -1
    min_value = sys.maxsize
    for k_index in range(len(fall_rates_measures)):
        current_distance = fall_rates_measures[k_index]
        if (current_distance < min_value) & (current_distance != 0):
            min_value = current_distance
            optimal_k = k_index

    # Display graphs for an optimal cluster amount
    print('Optimal clusters amount: ', optimal_k)

    centroids = place_centroids(points, optimal_k)
    clusters = []
    while True:
        new_clusters = assign_clusters_to_points(points, centroids)
        display_points(points, centroids, clusters)
        if np.array_equiv(clusters, new_clusters):
            break
        clusters = new_clusters
        new_centroids = move_centroids(points, clusters)
        centroids = new_centroids

    # K-means validation
    kmeans = KMeans(n_clusters=optimal_k)
    clusters = kmeans.fit_predict(points)
    display_points(points, centroids, assign_clusters_to_points(points, centroids))
