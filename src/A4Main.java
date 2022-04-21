package src;

import minet.example.mnist.MNISTDataset;
import minet.layer.*;
import minet.layer.init.WeightInitXavier;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.optim.Optimizer;
import minet.optim.SGD;
import minet.util.GradientChecker;
import minet.util.Pair;
import org.jblas.DoubleMatrix;
import org.jblas.util.Logger;

import java.io.IOException;
import java.util.*;

import src.EmbeddingBag;

public class A4Main {

    //TODO: maybe move the train and eval methods to another class.
    /**
     * Example A4Main class. Feel free to edit this file
     */
    public static void main(String[] args) throws IOException {

        if (args.length < 6) {
            System.out.println("Usage: java A4Main <part1/part2/part3/part4/part5> <seed> <trainFile> <devFile> <testFile> <vocabFile> <classesFile>");
            return;
        }

        boolean useTrainedWeights = false;

        if (args[0].equalsIgnoreCase("part3") || args[0].equalsIgnoreCase("part4") || args[0].equalsIgnoreCase("part5")) {
            useTrainedWeights = true;
        }

        // set jblas random seed (for reproducibility)
        int seed = Integer.parseInt(args[1]);
        org.jblas.util.Random.seed(seed);
        Random rnd = new Random(seed);

        // turn off jblas info messages
        Logger.getLogger().setLevel(Logger.WARNING);

        int batchsize = 50;
        int hiddimsEmbedding = 100;
        int hiddimsOthers = 200;
        // hyperparameters.
        double learningRate = 0.1;
        int maxEpochs = 500;
        int patience = 10;

        // load datasets
        System.out.println("\nLoading data...");
        VocabDataset trainset = new VocabDataset(batchsize, true, rnd, args[5], useTrainedWeights);
        trainset.fromFile(args[2]);

        VocabDataset devset = new VocabDataset(batchsize, false, rnd, args[5], false);
        devset.fromFile(args[3]);

        VocabDataset testset = new VocabDataset(batchsize, false, rnd, args[5], false);
        testset.fromFile(args[4]);

        System.out.printf("train: %d instances\n", trainset.getSize());
        System.out.printf("dev: %d instances\n", devset.getSize());
        System.out.printf("test: %d instances\n", testset.getSize());

        // create a network
        System.out.println("\nCreating network...");
        int indims = trainset.getInputDims();
        int outdims = 50;
        Sequential net;
        DoubleMatrix pretrainedWeights = trainset.getPretrainedWeights();

        // Determine if the network should print every step.
        boolean verbose = false;
        if (args[6] != null && args[6].equalsIgnoreCase("verbose")) {
            verbose = true;
        }


        GradientChecker gradientChecker = new GradientChecker();
        VocabClassifier vocabClassifier = new VocabClassifier(verbose);

        // TODO: redundant - make it all the same.
        CrossEntropy loss = new CrossEntropy();
        switch (args[0]) {
            case "part1":
                net = new Sequential(new Layer[]{
                        // Input to first hidden layer.
                        new Linear(indims, hiddimsEmbedding, new WeightInitXavier()),
                        new ReLU(),
                        // first to second hidden layer.
                        new Linear(hiddimsEmbedding, hiddimsOthers, new WeightInitXavier()),
                        new ReLU(),
                        // second to third hidden layer.
                        new Linear(hiddimsOthers, hiddimsOthers, new WeightInitXavier()),
                        new ReLU(),
                        // third hidden layer to output.
                        new Linear(hiddimsOthers, outdims, new WeightInitXavier()),
                        new Softmax()});

                vocabClassifier.trainAndEval(net, trainset, devset, testset, learningRate, maxEpochs, patience);
                break;

            case "part2":
                net = new Sequential(new Layer[]{
                        // Input to first hidden layer (Embedding bag).
                        new EmbeddingBag(indims, hiddimsEmbedding, new WeightInitXavier()),
                        new ReLU(),
                        // first to second hidden layer.
                        new Linear(hiddimsEmbedding, hiddimsOthers, new WeightInitXavier()),
                        new ReLU(),
                        // second to third hidden layer.
                        new Linear(hiddimsOthers, hiddimsOthers, new WeightInitXavier()),
                        new ReLU(),
                        // third hidden layer to output.
                        new Linear(hiddimsOthers, outdims, new WeightInitXavier()),
                        new Softmax()});

                vocabClassifier.trainAndEval(net, trainset, devset, testset, learningRate, maxEpochs, patience);
                break;

                //TODO: merge part 3, 4, and 5.
            case "part3":
                net = new Sequential(new Layer[]{
                        // Input to first hidden layer (Embedding bag). Use pretrained weights.
                        new EmbeddingBag(indims, hiddimsEmbedding, pretrainedWeights, false),
                        new ReLU(),
                        // first to second hidden layer.
                        new Linear(hiddimsEmbedding, hiddimsOthers, new WeightInitXavier()),
                        new ReLU(),
                        // second to third hidden layer.
                        new Linear(hiddimsOthers, hiddimsOthers, new WeightInitXavier()),
                        new ReLU(),
                        // third hidden layer to output.
                        new Linear(hiddimsOthers, outdims, new WeightInitXavier()),
                        new Softmax()});

                vocabClassifier.trainAndEval(net, trainset, devset, testset, learningRate, maxEpochs, patience);
                break;
            case "part4":
                net = new Sequential(new Layer[]{
                        // Input to first hidden layer (Embedding bag). Use pretrained weights.
                        // Freeze the weights - i.e. do not update them during training.
                        new EmbeddingBag(indims, hiddimsEmbedding, pretrainedWeights, true),
                        new ReLU(),
                        // first to second hidden layer.
                        new Linear(hiddimsEmbedding, hiddimsOthers, new WeightInitXavier()),
                        new ReLU(),
                        // second to third hidden layer.
                        new Linear(hiddimsOthers, hiddimsOthers, new WeightInitXavier()),
                        new ReLU(),
                        // third hidden layer to output.
                        new Linear(hiddimsOthers, outdims, new WeightInitXavier()),
                        new Softmax()});

                vocabClassifier.trainAndEval(net, trainset, devset, testset, learningRate, maxEpochs, patience);
                break;

            case "part5":
                // Changed dimensions due to the words2vec vocabulary having 300 dimensions.
                hiddimsEmbedding = 300;
                hiddimsOthers = 400;
                net = new Sequential(new Layer[]{
                        // Input to first hidden layer (Embedding bag). Use pretrained weights.
                        // Freeze the weights - i.e. do not update them during training.
                        new EmbeddingBag(indims, hiddimsEmbedding, pretrainedWeights, true),
                        new ReLU(),
                        // first to second hidden layer.
                        new Linear(hiddimsEmbedding, hiddimsOthers, new WeightInitXavier()),
                        new ReLU(),
                        // second to third hidden layer.
                        new Linear(hiddimsOthers, hiddimsOthers, new WeightInitXavier()),
                        new ReLU(),
                        // third hidden layer to output.
                        new Linear(hiddimsOthers, outdims, new WeightInitXavier()),
                        new Softmax()});

                vocabClassifier.trainAndEval(net, trainset, devset, testset, learningRate, maxEpochs, patience);
                break;
            default:
                System.out.println("Please select part1, part2, part3, part4 or part 5.");
        }
    }
}