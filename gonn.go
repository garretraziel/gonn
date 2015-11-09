package main

import (
    "flag"
    "fmt"
    "github.com/garretraziel/mnistloader"
    "github.com/garretraziel/matrices"
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
    images := make([]matrices.Matrix, len(imagesLoaded))
    for i, val := range imagesLoaded {
        images[i], err = matrices.InitMatrixWithValues(1, len(val), val)
        inputLength = len(val)
        if err != nil {
            panic(err)
        }
    }

    network := nn.InitNN([]int{inputLength, 30, 10})
    fmt.Println(network.Evaluate(images, labels))
}
