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
    labelsLoaded, err := mnistloader.ReadLabels(*labelPathPtr)
    if err != nil {
        panic(err)
    }
    imagesLoaded, err := mnistloader.ReadImages(*imagesPathPtr)
    if err != nil {
        panic(err)
    }

    labels, err := matrices.InitMatrixWithValues(1, len(labelsLoaded), labelsLoaded)
    if err != nil {
        panic(err)
    }
    images := make([]matrices.Matrix, len(imagesLoaded))
    for i, val := range imagesLoaded {
        images[i], err = matrices.InitMatrixWithValues(1, len(val), val)
        if err != nil {
            panic(err)
        }
    }
    _ = labels

    network := nn.InitNN([]int{4, 2, 3})
    fmt.Println(network)
    input, _ := matrices.InitMatrixWithValues(1, 4, []float64{1, 1, 1, 1})
    output := network.FeedForward(input)
    fmt.Println(output)
}
