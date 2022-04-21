package src;

import minet.layer.Sequential;

import java.util.ArrayList;
import java.util.List;

public class HyperparameterTuning {

    private Sequential net;

    // The hyperparameters to try.
    private List<Double> learningRatesToTry;
    private List<Integer> maxEpochsToTry;
    private List<Integer> patienceToTry;
    private VocabClassifier vocabClassifier;

    /**
     * Create a new hyperparameter tuning instance.
     * @param net
     * @param learningRatesToTry
     * @param maxEpochsToTry
     * @param patienceToTry
     */
    public HyperparameterTuning(Sequential net, VocabClassifier vocabClassifier, List<Double> learningRatesToTry, List<Integer> maxEpochsToTry, List<Integer> patienceToTry) {
        this.net = net;
        this.vocabClassifier = vocabClassifier;
        this.learningRatesToTry = learningRatesToTry;
        this.maxEpochsToTry = maxEpochsToTry;
        this.patienceToTry = patienceToTry;
    }

    /**
     * Use the randomized search method to find the best hyperparameters for the model.
     * It is very IMPORTANT to only use the validation set to determine hyperparameters -> avoid overfitting.
     * @param iterations
     */
    public void randomizedSearch(int iterations, VocabDataset trainset, VocabDataset devset, VocabDataset testset, boolean verbose) {
        // Store the best validation accuracy found so far.
        double bestAccuracy = 0;

        // Store the variable values of the best run so far.
        double learningRateOnBestResult = 0;
        int maxEpochsOnBestResult = 0;
        int patienceOnBestResult = 0;

        System.out.println("HYPERPARAMETER TUNING STARTING");

        // Try out different combinations of hyperparameters.
        for (int i=0; i<iterations; i++) {
            // Make a temporary net with the layers of the passed in net.
            Sequential tempNet = new Sequential(net);

            // Get random index of value to try for each hyperparameter.
            double randomLearningRate = learningRatesToTry.get(getRandomIndex(learningRatesToTry.size()));
            int randomMaxEpochs = maxEpochsToTry.get(getRandomIndex(maxEpochsToTry.size()));
            int randomPatience = patienceToTry.get(getRandomIndex(patienceToTry.size()));

            System.out.println("\nIteration: "+ i);
//            if (verbose) {
                System.out.println("HYPER-PARAMETERS RANDOMLY SELECTED\nlearningRate: " + randomLearningRate+ ", maxEpochs: "+ randomMaxEpochs+ ", patience: "+ randomPatience);
//            }
            // Get best validation accuracy of the network using the randomly selected hyperparameter values.
            double valAcc = vocabClassifier.tuningProcess(tempNet, trainset, devset, randomLearningRate, randomMaxEpochs, randomPatience);

            if (valAcc > bestAccuracy) {
                bestAccuracy = valAcc; // update the best accuracy value.

                // update the values of the hyperparameters to the new ones -> they are the ones that produces the best results.
                learningRateOnBestResult = randomLearningRate;
                maxEpochsOnBestResult = randomMaxEpochs;
                patienceOnBestResult = randomPatience;
            }
        }

        Sequential finalNet = new Sequential(net);
        System.out.println("\n\nFINISHED TUNING");
        System.out.println("BEST HYPER-PARAMETERS FOUND \n learningRate: " + learningRateOnBestResult + ", maxEpochs: "+ maxEpochsOnBestResult + ", patience: "+ patienceOnBestResult);
        vocabClassifier.trainAndEval(finalNet, trainset, devset, testset, learningRateOnBestResult, maxEpochsOnBestResult, patienceOnBestResult);
    }

    public int getRandomIndex(int listSize) {
        return (int)(Math.random() * listSize);
    }



}
