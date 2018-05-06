import numpy as np
import math
import matplotlib.pyplot as plt

# FUNGSI UNTUK LOAD DATA
def loadData(path) :
    data_train = []
    data_train_file = open(path)
    for lines in data_train_file:
        num_list = []
        for number in lines.strip().split():
            num_list.append(float(number))
        data_train.append(num_list)
    return data_train

# FUNGSI UNTUK MENGHITUNG EUCLIDEAN DISTANCE
def euclidean(data1,data2) :
    x1,y1 = data1[0],data1[1]
    x2,y2 = data2[0],data2[1]
    return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))

# FUNGSI UNTUK MENGHITUNG SSE
def SSE(centroids,dataset) :
    euclideanSum = [0.0 for i in range(len(centroids))]
    for data in dataset :
        euclideanSum[int(data[2])] += pow(euclidean(data,centroids[int(data[2])]),2)
    return sum(euclideanSum)

# FUNGSI UNTUK MEMBUAT SCATTER PLOT
def scatterPlot(data,centroids,title) :
    plt.title(title)
    plt.scatter(np.asarray(data)[:, 0], np.asarray(data)[:, 1], c=np.asarray(data)[:, 2])
    plt.scatter(np.asarray(centroids)[:, 0], np.asarray(centroids)[:, 1], c='r')
    plt.show()

# FUNGSI UNTUK MELAKUKAN CLUSTERING DATA
def KMeans(dataTrain,dataTest,k=5) :

    # RANDOM CENTROID AWAL
    centroids = [[np.random.randint(0, 35), np.random.randint(0, 35)] for i in range(k)]

    # TRESHOLD
    T = 0.5

    # VARIABEL UNTUK MENAMPUNG NILAI SSE SETIAP ITERASI
    SSE_old,SSE_new = 0.0,0.0

    while True :
        TrainLabels = []

        # MENENTUKAN CENTROID SETIAP DATA DENGAN MENGHITUNG EUCLIDEAN DISTANCE
        for data in dataTrain:
            distances = [euclidean(centroids[i], data) for i in range(k)]
            TrainLabels.append(distances.index(min(distances)))
        labeledDataTrain = np.concatenate((dataTrain,np.expand_dims(TrainLabels,axis=0).T),axis=1)

        # MENENTUKAN CENTROID BARU DENGAN MENGHITUNG RATA-RATA DATA DI SETIAP CENTROID
        newCentroids = [[0.0,0.0] for i in range (k)]
        n = [0 for i in range(k)]
        for data in labeledDataTrain :
            newCentroids[int(data[2])][0] += data[0]
            newCentroids[int(data[2])][1] += data[1]
            n[int(data[2])] += 1

        for i in range(k) :
            if n[i] != 0 :
                centroids[i] = [newCentroids[i][0]/n[i],newCentroids[i][1]/n[i]]

        SSE_new = SSE(centroids,labeledDataTrain)

        if (abs(SSE_new-SSE_old)) > T :
            SSE_old = SSE_new
        else:
            break

    # MENAMPILKAN SCATTER PLOT DATA TRAIN YANG SUDAH DIKLASTERISASI
    scatterPlot(labeledDataTrain,centroids,"DATA TRAIN (K=%i)"%k)

    # MELAKUKAN CLUSTERING DATA TEST
    TestLabels = []
    for data in dataTest:
        distances = [euclidean(centroids[i], data) for i in range(k)]
        TestLabels.append(distances.index(min(distances)))
    labeledDataTest = np.concatenate((dataTest, np.expand_dims(TestLabels, axis=0).T), axis=1)
    # MENAMPILKAN SCATTER PLOT DATA TEST YANG SUDAH DI KLASTERISASI
    scatterPlot(labeledDataTest,centroids,"DATA TEST (K=%i)"%k)

    # MENAMPILKAN SSE TRAINING DAN SSE TESTING
    print("K =",k," SSE Training =",SSE(centroids,labeledDataTrain))
    print("K =",k," SSE Testing =",SSE(centroids,labeledDataTest))

    # MENYIMPAN LABEL KE DALAM FILE
    np.savetxt('TrainsetLabel.txt',[TrainLabels],fmt='%i',delimiter=',')
    np.savetxt('TestsetLabel.txt',[TestLabels],fmt='%i',delimiter=',')

    return [SSE(centroids,labeledDataTrain),SSE(centroids,labeledDataTest)]

if __name__ == '__main__':

    dataTrain = loadData('./Tugas 2/TrainsetTugas2.txt')
    dataTest = loadData('./Tugas 2/TestsetTugas2.txt')


    SSEarr = []
    # ITERASI DARI K=1 SAMPAI K=10
    for i in range(1,11) :
        SSEarr.append(KMeans(dataTrain,dataTest,k=i))

    xlabel = ["k=%i"%i for i in range(1,11)]
    x = [i for i in range(len(xlabel))]

    # MENAMPILKAN GRAFIK SSE DATA TRAIN
    plt.title("DATA TRAIN")
    plt.xticks(x,xlabel)
    plt.xlabel("n_clusters (K)")
    plt.ylabel("SSE")
    plt.plot(np.asarray(SSEarr)[:,0],'b--o')
    plt.show()

    # MENAMPILKAN GRAFIK SSE DATA TEST
    plt.title("DATA TEST")
    plt.xticks(x, xlabel)
    plt.xlabel("n_clusters (K)")
    plt.ylabel("SSE")
    plt.plot(np.asarray(SSEarr)[:,1],'b--o')
    plt.show()
