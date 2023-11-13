import pygame
import random
import numpy as np


MAX_DISTANCE = 50


class Point:
    x = 0
    y = 0
    is_green = False
    is_yellow = False
    is_red = False

    def __init__(self, x, y):
        self.x = x
        self.y = y


def dist(first_point, second_point):
    # Считаем расстояние между точками
    return np.sqrt((first_point.x - second_point.x)**2 + (first_point.y - second_point.y)**2)


def near_points(point):
    count = random.randint(2, 5)
    points = []
    for i in range(count):
        x = random.randint(-20, 20)
        y = random.randint(-20, 20)
        points.append((point[0] + x, point[1] + y))

    return points


def make_flags(points):
    for point in points:
        point.is_red = False
        point.is_green = False
        point.is_yellow = False

    for current_point in points:
        neighbours = 0
        for near_point in points:
            if current_point.x == near_point.x and current_point.y == near_point.y:
                continue

            if dist(current_point, near_point) <= MAX_DISTANCE:
                neighbours += 1

        if neighbours >= 2:
            current_point.is_green = True

    for current_point in points:
        green_neighbours = 0
        for near_point in points:
            if current_point.x == near_point.x and current_point.y == near_point.y:
                continue

            if dist(current_point, near_point) <= MAX_DISTANCE and near_point.is_green:
                green_neighbours += 1

        if green_neighbours == 1:
            current_point.is_yellow = True
        elif green_neighbours == 0:
            current_point.is_red = True

    screen.fill(color='#FFFFFF')
    pygame.display.update()

    for point in points:
        if point.is_green:
            pygame.draw.circle(screen, color='green', center=(point.x, point.y), radius=5)
        elif point.is_yellow:
            pygame.draw.circle(screen, color='yellow', center=(point.x, point.y), radius=5)
        elif point.is_red:
            pygame.draw.circle(screen, color='red', center=(point.x, point.y), radius=5)
    pygame.display.update()

    return points


def make_clusters(points):
    visited_points = set()
    hermits = []
    clusters = []
    for hermit in points:
        if hermit.is_red:
            hermits.append(hermit)
            visited_points.add(hermit)

    while len(visited_points) != len(points):
        random_point = points[random.randint(0, len(points) - 1)]
        while random_point.is_red:
            random_point = points[random.randint(0, len(points) - 1)]

        while random_point.is_yellow:
            random_point = points[random.randint(0, len(points) - 1)]

        while random_point in visited_points:
            random_point = points[random.randint(0, len(points) - 1)]

        visited_points.add(random_point)
        cluster= [random_point]
        neighbours = []
        for point in points:
            if point in visited_points:
                continue

            if point.is_red:
                continue

            if dist(random_point, point) <= MAX_DISTANCE:
                if point.is_yellow:
                    cluster.append(point)
                    visited_points.add(point)
                else:
                    neighbours.append(point)

        while len(neighbours) != 0:
            random_neighbour = neighbours.pop(random.randint(0, len(neighbours) - 1))
            visited_points.add(random_neighbour)
            cluster.append(random_neighbour)

            for point in points:
                if point in visited_points:
                    continue

                if point.is_red:
                    continue

                if dist(random_neighbour, point) <= MAX_DISTANCE:
                    if point.is_yellow:
                        cluster.append(point)
                        visited_points.add(point)
                    else:
                        neighbours.append(point)

        clusters.append(cluster)

    screen.fill(color='#FFFFFF')
    colors = ['black', 'gray', 'brown', 'orange', 'lime', 'cyan', 'blue', 'navy',
              'magenta', 'purple', 'violet', 'pink']
    for cluster in clusters:
        color = colors.pop(random.randint(0, len(colors) - 1))
        for point in cluster:
            pygame.draw.circle(screen, color=color, center=(point.x, point.y), radius=5)

    for hermit in hermits:
        pygame.draw.circle(screen, color='red', center=(hermit.x, hermit.y), radius=5)

    pygame.display.update()
    return points


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill(color='#FFFFFF')
    pygame.display.update()

    is_active = True
    is_moustbottomdown = False
    points = []
    while is_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_active = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_moustbottomdown = True
                    center_coordinates = event.pos
                    point = Point(center_coordinates[0], center_coordinates[1])
                    points.append(point)
                    pygame.draw.circle(screen, color='black', center=center_coordinates, radius=5)
                    pygame.display.update()
            if event.type == pygame.MOUSEBUTTONUP:
                is_moustbottomdown = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_c:
                    points = []
                    screen.fill(color='#FFFFFF')
                    pygame.display.update()
                if event.key == pygame.K_SPACE:
                    points = make_flags(points)
                if event.key == pygame.K_RETURN:
                    points = make_clusters(points)
            if event.type == pygame.MOUSEMOTION and is_moustbottomdown:
                new_point = Point(event.pos[0], event.pos[1])
                if dist(new_point, points[-1]) > 20:
                    center_coordinates = event.pos
                    pygame.draw.circle(screen, color='black', center=center_coordinates, radius=5)
                    point = Point(center_coordinates[0], center_coordinates[1])
                    points.append(point)
                    random_points = near_points(center_coordinates)

                    for coords in random_points:
                        pygame.draw.circle(screen, color='black', center=coords, radius=5)
                        point_to_append = Point(coords[0], coords[1])
                        points.append(point_to_append)

                    pygame.display.update()