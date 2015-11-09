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
    labels, err := mnistloader.ReadLabels(*labelPathPtr)
    if err != nil {
        panic(err)
    }
    imagesLoaded, err := mnistloader.ReadImages(*imagesPathPtr)
    if err != nil {
        panic(err)
    }

    inputLength := 0
    inputs := make([]nn.TrainItem, len(imagesLoaded))
    for i, val := range imagesLoaded {
        inputs[i], err = nn.InitTrainItem(val, labels[i])
        if err != nil {
            panic(err)
        }
    }

    network := nn.InitNN([]int{inputLength, 30, 10})
    network.Train(inputs, 30, 10, 3.0)
}
