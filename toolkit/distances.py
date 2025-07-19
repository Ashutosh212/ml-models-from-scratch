import numpy as np

def SquaredEuclideanDistance(data1, data2):
    distance = np.sum(np.square(np.subtract(data1, data2)))
    return distance

def main():

    dimn = 2
    data1 = np.random.choice(np.arange(1,10), (2,dimn))
    data2 = np.random.choice(np.arange(1,10), (2,dimn))
    print(f"data1: {data1}")
    print(f"data2: {data2}")
    print(SquaredEuclideanDistance(data1=data1, data2=data2))

if __name__=="__main__":
    main()