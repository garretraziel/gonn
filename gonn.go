package main

import (
    "flag"
    "github.com/garretraziel/mnistloader"
    "github.com/garretraziel/nn"
)

func main() {
    labelPathPtr := flag.String("labels", "dataset/train-labels.idx1-ubyte", "path to MNIST labels file")
    imagesPathPtr := flag.String("images", "dataset/train-images.idx3-ubyte", "path to MNIST images file")
    flag.Parse()
    labels, distinct, err := mnistloader.ReadLabels(*labelPathPtr)
    if err != nil {
        panic(err)
    }
    imagesLoaded, inputLength, err := mnistloader.ReadImages(*imagesPathPtr)
    if err != nil {
        panic(err)
    }

    inputs := make([]nn.TrainItem, len(imagesLoaded))
    for i, val := range imagesLoaded {
        inputs[i], err = nn.InitTrainItem(val, labels[i], distinct)
        if err != nil {
            panic(err)
        }
    }

    testData := inputs[len(inputs) - 10000:]
    inputs = inputs[:len(inputs) - 10000]

    network := nn.InitNN([]int{inputLength, 100, distinct})
    network.Train(inputs, 60, 10, 0.1, 5.0, testData, false)
}
