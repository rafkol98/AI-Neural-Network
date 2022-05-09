package src;

import minet.layer.Layer;
import minet.layer.Sequential;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.optim.Optimizer;
import minet.optim.SGD;
import minet.util.Pair;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class VocabClassifier {

    private boolean verbose;

    public VocabClassifier(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * Train the model and return the best validation accuracy found. Used for randomizedSearch tuning procedure.
     *
     * @param net the passed in network.
     * @param trainset the training set.
     * @param devset the development/validation set.
     * @param learningRate the specified learning rate.
     * @param maxEpochs the number of max epochs.
     * @param patience the patience value.
     * @return
     */
    public double tuningProcess(Sequential net, VocabDataset trainset, VocabDataset devset, double learningRate, int maxEpochs, int patience) {
        CrossEntropy loss = new CrossEntropy();
        Optimizer sgd = new SGD(net, learningRate);

        double bestValAcc = train(net, loss, sgd, trainset, devset, maxEpochs, patience);
        return bestValAcc;
    }


    /**
     * Train and evaluate the algorithm.
     *
     * @param net the passed in network.
     * @param trainset the training set.
     * @param devset the development/validation set.
     * @param testset the testing set.
     * @param learningRate the specified learning rate.
     * @param maxEpochs the number of max epochs.
     * @param patience the patience value.
     */
    public void trainAndEval(Sequential net, VocabDataset trainset, VocabDataset devset, VocabDataset testset, double learningRate, int maxEpochs, int patience) {
        CrossEntropy loss = new CrossEntropy();
        Optimizer sgd = new SGD(net, learningRate);
        System.out.println(net);

        // train network
        System.out.println("\nTraining...");

        double bestValAcc = train(net, loss, sgd, trainset, devset, maxEpochs, patience);

        // perform on test set
        double testAcc = eval(net, testset);
        System.out.printf("\nTest accuracy: %.4f\n", testAcc);
    }


    /**
     * Convert a mini-batch of the vocabulary dataset to data structure that can be used by the network - pair
     * of double matrices, where the first element of the pair is the input (X) and the second is the label (Y).
     *
     * @param batch
     * @return two DoubleMatrix objects: X (input) and Y (labels)
     */
    public Pair<DoubleMatrix, DoubleMatrix> convertToDoubleMatrixPair(List<Pair<double[], Integer>> batch) {
        if (batch == null)
            return null;

        double[][] xs = new double[batch.size()][];
        double[] ys = new double[batch.size()];

        // iterate through the batch size.
        for (int i = 0; i < batch.size(); i++) {
            xs[i] = batch.get(i).first;
            ys[i] = (double) batch.get(i).second;
        }
        // create new double matrices for X and Y.
        DoubleMatrix X = new DoubleMatrix(xs);
        DoubleMatrix Y = new DoubleMatrix(ys.length, 1, ys);
        return new Pair<DoubleMatrix, DoubleMatrix>(X, Y);
    }

    /**
     * calculate classification accuracy of an ANN for our NLP problem.
     *
     * @param net  an ANN model
     * @param data the vocabulary dataset
     * @return the classification accuracy value (double, in the range of [0,1])
     */
    public double eval(Layer net, VocabDataset data) {
        // reset index of the data
        data.reset();

        // the number of correct predictions so far
        double correct = 0;

        while (true) {
            // we evaluate per mini-batch
            Pair<DoubleMatrix, DoubleMatrix> batch = convertToDoubleMatrixPair(data.getNextMiniBatch());
            if (batch == null)
                break;

            // Perform forward pass to calculate the predictions.
            DoubleMatrix Yhat = net.forward(batch.first);

            // Assign prediction with the highest probability.
            int[] preds = Yhat.rowArgmaxs();

            // Count the number of correct predictions.
            for (int i = 0; i < preds.length; i++) {
                if (preds[i] == (int) batch.second.data[i])
                    correct++;
            }
        }

        // Calculate the classification accuract.
        double acc = correct / data.getSize();
        return acc;
    }


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
    public double train(Layer net, Loss loss, Optimizer optimizer, VocabDataset traindata,
                        VocabDataset devdata, int nEpochs, int patience) {
        long startTime = System.nanoTime(); // start timer.

        List<Double> trainingAccuracies = new ArrayList<>();
        List<Double> validationAccuracies = new ArrayList<>();

        int notAtPeak = 0;  // the number of times not at peak
        double peakAcc = -1;  // the best accuracy of the previous epochs
        double totalLoss = 0;  // the total loss of the current epoch

        traindata.reset(); // reset index and shuffle the dataset before training

        for (int e = 0; e < nEpochs; e++) {
            totalLoss = 0;

            while (true) {
                // get the next mini-batch
                Pair<DoubleMatrix, DoubleMatrix> batch = convertToDoubleMatrixPair(traindata.getNextMiniBatch());

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
            // add trainAcc to the ArrayList storing all accuracies.
            trainingAccuracies.add(trainAcc);

            // add valAcc to the ArrayList storing all validation accuracies.
            double valAcc = eval(net, devdata);
            validationAccuracies.add(valAcc);

            if (verbose) {
                System.out.printf("epoch: %4d\tloss: %5.4f\ttrain-accuracy: %3.4f\tdev-accuracy: %3.4f\n", e, totalLoss, trainAcc, valAcc);
            }

            // check termination condition
            if (valAcc <= peakAcc) {
                notAtPeak += 1;
                if (verbose) {
                    System.out.printf("not at peak %d times consecutively\n", notAtPeak);
                }
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
        System.out.println("Best Training Accuracy: " + Collections.max(trainingAccuracies));
        System.out.println("Best Validation Accuracy: " + Collections.max(validationAccuracies));
        System.out.println("Total training time: " + trainingDuration + " seconds");

        // return best validation accuracy - used for hyperparameter tuning.
        return Collections.max(validationAccuracies);
    }
}
