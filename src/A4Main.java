package src;

import minet.example.mnist.MNISTDataset;
import minet.layer.*;
import minet.layer.init.WeightInitXavier;
import minet.loss.CrossEntropy;
import minet.optim.Optimizer;
import minet.optim.SGD;
import org.jblas.util.Logger;

import java.io.IOException;
import java.util.Random;
import src.EmbeddingBag;

public class A4Main {

    /**
     * Example A4Main class. Feel free to edit this file 
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 6){
            System.out.println("Usage: java A4Main <part1/part2/part3/part4> <seed> <trainFile> <devFile> <testFile> <vocabFile> <classesFile>");
            return;
        }        

        // set jblas random seed (for reproducibility)
        int seed = Integer.parseInt(args[1]);
		org.jblas.util.Random.seed(seed);
		Random rnd = new Random(seed);
		
        // turn off jblas info messages
        Logger.getLogger().setLevel(Logger.WARNING);


        double learningRate = 0.1;
        int batchsize = 50;
        int maxEpochs = 500;
        int patience = 10;
        int hiddimsFirst = 100;
        int hiddimsOthers = 200;


        // load datasets
        System.out.println("\nLoading data...");
        MNISTDataset trainset = new MNISTDataset(batchsize, true, rnd);
        trainset.fromFile(args[1]);
        MNISTDataset devset = new MNISTDataset(batchsize, false, rnd);
        devset.fromFile(args[2]);
        MNISTDataset testset = new MNISTDataset(batchsize, false, rnd);
        testset.fromFile(args[3]);

        System.out.printf("train: %d instances\n", trainset.getSize());
        System.out.printf("dev: %d instances\n", devset.getSize());
        System.out.printf("test: %d instances\n", testset.getSize());

        // create a network
        System.out.println("\nCreating network...");
        int indims = trainset.getInputDims();
        int outdims = 10;
        Sequential net = new Sequential(new Layer[] {
                // Input to first hidden layer.
                new Linear(indims, hiddimsFirst, new WeightInitXavier()),
                new ReLU(),
                // first to second hidden layer.
                new Linear(hiddimsFirst, hiddimsOthers, new WeightInitXavier()),
                new ReLU(),
                // second to third hidden layer.
                new Linear(hiddimsOthers, hiddimsOthers, new WeightInitXavier()),
                new ReLU(),
                // third hidden layer to output.
                new Linear(hiddimsOthers, outdims, new WeightInitXavier()),
                new Softmax()});

        CrossEntropy loss = new CrossEntropy();
        Optimizer sgd = new SGD(net, learningRate);
        System.out.println(net);
    }
}
