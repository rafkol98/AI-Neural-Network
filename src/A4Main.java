package src;

import minet.example.mnist.MNISTDataset;
import minet.layer.*;
import minet.layer.init.WeightInitXavier;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.optim.Optimizer;
import minet.optim.SGD;
import minet.util.Pair;
import org.jblas.DoubleMatrix;
import org.jblas.util.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import src.EmbeddingBag;

public class A4Main {

    /**
     * Example A4Main class. Feel free to edit this file
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 6) {
            System.out.println("Usage: java A4Main <part1/part2/part3/part4> <seed> <trainFile> <devFile> <testFile> <vocabFile> <classesFile>");
            return;
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

        // load datasets
        System.out.println("\nLoading data...");
        VocabDataset trainset = new VocabDataset(batchsize, true, rnd);
        trainset.fromFile(args[2], args[5]);

        VocabDataset devset = new VocabDataset(batchsize, false, rnd);
        devset.fromFile(args[3], args[5]);

        VocabDataset testset = new VocabDataset(batchsize, false, rnd);
        testset.fromFile(args[4], args[5]);

        System.out.printf("train: %d instances\n", trainset.getSize());
        System.out.printf("dev: %d instances\n", devset.getSize());
        System.out.printf("test: %d instances\n", testset.getSize());

        // create a network
        System.out.println("\nCreating network...");
        int indims = trainset.getInputDims();
        int outdims = 50;
        Sequential net;

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

                trainAndEval(net, trainset, devset, testset);
                break;
            case "part2":
                net = new Sequential(new Layer[]{
                        // Input to first hidden layer.
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

                trainAndEval(net, trainset, devset, testset);
                break;
            default:
                System.out.println("Please select part1, part2, part3 or part4.");
        }
    }

    public static void trainAndEval(Sequential net, VocabDataset trainset, VocabDataset devset, VocabDataset testset) {
        double learningRate = 2;
        int maxEpochs = 500;
        int patience = 10;

        CrossEntropy loss = new CrossEntropy();
        Optimizer sgd = new SGD(net, learningRate);
        System.out.println(net);

        // train network
        System.out.println("\nTraining...");

        train(net, loss, sgd, trainset, devset, maxEpochs, patience);

        // perform on test set
        double testAcc = eval(net, testset);
        System.out.printf("\nTest accuracy: %.4f\n", testAcc);
    }


    /**
     * Convert a mini-batch of the vocabulary dataset to data structure that can be used by the network
     *
     * @param batch
     * @return two DoubleMatrix objects: X (input) and Y (labels)
     */
    public static Pair<DoubleMatrix, DoubleMatrix> fromBatch(List<Pair<double[], Integer>> batch) {
        if (batch == null)
            return null;

        double[][] xs = new double[batch.size()][];
        double[] ys = new double[batch.size()];
        for (int i = 0; i < batch.size(); i++) {
            xs[i] = batch.get(i).first;


            ys[i] = (double) batch.get(i).second;

        }
        DoubleMatrix X = new DoubleMatrix(xs);
        DoubleMatrix Y = new DoubleMatrix(ys.length, 1, ys);
        return new Pair<DoubleMatrix, DoubleMatrix>(X, Y);
    }

//    /**
//     * Convert a mini-batch of the vocabulary dataset to data structure that can be used by the network
//     *
//     * @param batch a list of MNIST items, each of which is a pair of (input image, output label)
//     * @return two DoubleMatrix objects: X (input) and Y (labels)
//     */
//    public static List<int[]> fromBatch(List<Pair<double[], Integer>> batch) {
//        if (batch == null)
//            return null;
//
//        List<int[]> wordsIndices = new ArrayList<>();
//
//
//        double[][] xs = new double[batch.size()][];
//        double[] ys = new double[batch.size()];
//        for (int i = 0; i < batch.size(); i++) {
//            xs[i] = batch.get(i).first;
//            ys[i] = (double) batch.get(i).second;
//        }
//        DoubleMatrix X = new DoubleMatrix(xs);
//        DoubleMatrix Y = new DoubleMatrix(ys.length, 1, ys);
//        return new Pair<DoubleMatrix, DoubleMatrix>(X, Y);
//    }

    //TODO: change!

    /**
     * calculate classification accuracy of an ANN on a given dataset.
     *
     * @param net  an ANN model
     * @param data the vocabulary dataset
     * @return the classification accuracy value (double, in the range of [0,1])
     */
    public static double eval(Layer net, VocabDataset data) {
        // reset index of the data
        data.reset();

        // the number of correct predictions so far
        double correct = 0;

        while (true) {
            // we evaluate per mini-batch
            Pair<DoubleMatrix, DoubleMatrix> batch = fromBatch(data.getNextMiniBatch());
            if (batch == null)
                break;

            // perform forward pass to compute Yhat (the predictions)
            // each row of Yhat is a probabilty distribution over 10 digits
            DoubleMatrix Yhat = net.forward(batch.first);

            // the predicted digit for each image is the one with the highest probability
            int[] preds = Yhat.rowArgmaxs();

            // count how many predictions are correct
            for (int i = 0; i < preds.length; i++) {
                if (preds[i] == (int) batch.second.data[i])
                    correct++;
            }
        }

        // compute classification accuracy
        double acc = correct / data.getSize();
        return acc;
    }

    //TODO: change!

    /**
     * train an ANN for our NLP problem.
     *
     * @param net       an ANN model to be trained
     * @param loss      a loss function object
     * @param optimizer the optimizer used for updating the model's weights (currently only SGD is supported)
     * @param traindata training dataset
     * @param devdata   validation dataset (also called development dataset), used for early stopping
     * @param nEpochs   the maximum number of training epochs
     * @param patience  the maximum number of consecutive epochs where validation performance is allowed to non-increased, used for early stopping
     */
    public static void train(Layer net, Loss loss, Optimizer optimizer, VocabDataset traindata,
                             VocabDataset devdata, int nEpochs, int patience) {
        long startTime = System.nanoTime(); // start timer.

        int notAtPeak = 0;  // the number of times not at peak
        double peakAcc = -1;  // the best accuracy of the previous epochs
        double totalLoss = 0;  // the total loss of the current epoch

        traindata.reset(); // reset index and shuffle the dataset before training


        for (int e = 0; e < nEpochs; e++) {
            totalLoss = 0;

            while (true) {
                // get the next mini-batch
                Pair<DoubleMatrix, DoubleMatrix> batch = fromBatch(traindata.getNextMiniBatch());

                if (batch == null)
                    break;

                // always reset the gradients before performing backward
                optimizer.resetGradients();
                // calculate the loss value
                DoubleMatrix Yhat = net.forward(batch.first);
                double lossVal = loss.forward(batch.second, Yhat);

                // calculate gradients of the weights using backprop algorithm
                net.backward(loss.backward());

                // update the weights using the calculated gradients
                optimizer.updateWeights();

                totalLoss += lossVal;
            }

            // evaluate and print performance
            double trainAcc = eval(net, traindata);
            double valAcc = eval(net, devdata);
            System.out.printf("epoch: %4d\tloss: %5.4f\ttrain-accuracy: %3.4f\tdev-accuracy: %3.4f\n", e, totalLoss, trainAcc, valAcc);

            // check termination condition
            if (valAcc <= peakAcc) {
                notAtPeak += 1;
                System.out.printf("not at peak %d times consecutively\n", notAtPeak);
            } else {
                notAtPeak = 0;
                peakAcc = valAcc;
            }
            if (notAtPeak == patience)
                break;
        }

        long endTime = System.nanoTime(); // stop timer.

        // calculate training duration - divide by 10^-9 to convert ns to seconds.
        long trainingDuration = (endTime - startTime) / 1000000000;

        System.out.println("\ntraining is finished");
        System.out.println("Total training time: " + trainingDuration + " seconds");
    }

}