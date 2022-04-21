package src;

import minet.layer.Sequential;

import java.util.ArrayList;

public class HyperparameterTuning {

    private Sequential net;

    // The hyperparameters to try.
    private ArrayList<Double> learningRatesToTry;
    private ArrayList<Integer> maxEpochsToTry;
    private ArrayList<Integer> patienceToTry;
    private VocabClassifier vocabClassifier;

    /**
     * Create a new hyperparameter tuning instance.
     * @param net
     * @param learningRatesToTry
     * @param maxEpochsToTry
     * @param patienceToTry
     */
    public HyperparameterTuning(Sequential net, VocabClassifier vocabClassifier, ArrayList<Double> learningRatesToTry, ArrayList<Integer> maxEpochsToTry, ArrayList<Integer> patienceToTry) {
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
    public void randomizedSearch(int iterations, VocabDataset trainset, VocabDataset devset, VocabDataset testset) {
        // Store the best validation accuracy found so far.
        double bestAccuracy = 0;

        // Store the variable values of the best run so far.
        double learningRateOnBestResult = 0;
        int maxEpochsOnBestResult = 0;
        int patienceOnBestResult = 0;

        // Try out different combinations of hyperparameters.
        for (int i=0; i<iterations; i++) {
            // Get random index of value to try for each hyperparameter.
            double randomLearningRate = learningRatesToTry.get(getRandomIndex(learningRatesToTry.size()));
            int randomMaxEpochs = maxEpochsToTry.get(getRandomIndex(maxEpochsToTry.size()));
            int randomPatience = patienceToTry.get(getRandomIndex(patienceToTry.size()));

            System.out.println("Iteration: "+ i);
            System.out.println("HYPER-PARAMETERS RANDOMLY SELECTED\n learningRate: " + randomLearningRate+ ", maxEpochs: "+ randomMaxEpochs+ ", patience: "+ randomPatience);
            // Get best validation accuracy of the network using the randomly selected hyperparameter values.
            double valAcc = vocabClassifier.tuningProcess(net, trainset, devset, randomLearningRate, randomMaxEpochs, randomPatience);

            if (valAcc > bestAccuracy) {
                bestAccuracy = valAcc; // update the best accuracy value.

                // update the values of the hyperparameters to the new ones -> they are the ones that produces the best results.
                learningRateOnBestResult = randomLearningRate;
                maxEpochsOnBestResult = randomMaxEpochs;
                patienceOnBestResult = randomPatience;
            }
        }

        //TODO: calculate testing accuracy of best model.
        System.out.println("\n\n BEST MODEL AFTER TUNING");
        System.out.println("BEST HYPER-PARAMETERS FOUND \n learningRate: " + learningRateOnBestResult + ", maxEpochs: "+ maxEpochsOnBestResult + ", patience: "+ patienceOnBestResult);
        vocabClassifier.trainAndEval(net, trainset, devset, testset, learningRateOnBestResult, maxEpochsOnBestResult, patienceOnBestResult);
    }

    public int getRandomIndex(int listSize) {
        return (int)(Math.random() * listSize);
    }



}
