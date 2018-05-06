import numpy as np
import math
import matplotlib.pyplot as plt

def euclidean(data1,data2) :
    x1,y1 = data1[0],data1[1]
    x2,y2 = data2[0],data2[1]
    return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))

dataTrain = []
dataset_file = open('TrainsetTugas2.txt')
for lines in dataset_file:
    num_list = []
    for number in lines.strip().split():
        num_list.append(float(number))
    num_list.append(-1)
    dataTrain.append(num_list)

dataTest = []
dataset_file = open('TestsetTugas2.txt')
for lines in dataset_file:
    num_list = []
    for number in lines.strip().split():
        num_list.append(float(number))
    num_list.append(-1)
    dataTest.append(num_list)

k=4
centroids = [[np.random.randint(0, 35), np.random.randint(0, 35)] for i in range(k)]

for x in range(10):

    for data in dataTrain:
        distances = [euclidean(centroids[i], data) for i in range(k)]
        minDistance = distances.index(min(distances))
        data[2] = (int(minDistance))

    newCentroids = []
    for i in range(k):
        n = 0
        sumX, sumY = 0.0, 0.0
        for data in dataTrain:
            if data[2] == i:
                sumX += data[0]
                sumY += data[1]
                n += 1
        if n > 0:
            newCentroids.append([sumX / n, sumY / n])
    k = len(newCentroids)

    if -0.001 < np.mean(newCentroids) - np.mean(centroids) < 0.001:
        break
    else:
        centroids = newCentroids

plt.scatter(np.asarray(dataTrain)[:, 0], np.asarray(dataTrain)[:, 1], c=np.asarray(dataTrain)[:, 2])
plt.scatter(np.asarray(newCentroids)[:, 0], np.asarray(newCentroids)[:, 1], c='r')
plt.show()

for data in dataTest:
    distances = [euclidean(centroids[i], data) for i in range(k)]
    minDistance = distances.index(min(distances))
    data[2] = (int(minDistance))

plt.scatter(np.asarray(dataTest)[:, 0], np.asarray(dataTest)[:, 1], c=np.asarray(dataTest)[:, 2])
plt.show()
