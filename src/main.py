import mnist_loader
import network2

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 20,20,10], cost=network2.CrossEntropyCost)
    net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data, 
    monitor_evaluation_accuracy=True)