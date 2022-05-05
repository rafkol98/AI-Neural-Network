package src;

import minet.layer.*;
import minet.layer.init.WeightInitXavier;

import java.util.ArrayList;
import java.util.List;

public class HyperparameterTuning {

    // The hyperparameters to try.
    private List<Double> learningRatesToTry;
    private List<Integer> maxEpochsToTry;
    private List<Integer> patienceToTry;
    private VocabClassifier vocabClassifier;
    private int indims, hiddimsEmbedding, hiddimsOthers, outdims;
    private boolean linearNetwork;



    /**
     * Create a new hyperparameter tuning instance.
     *
     * @param linearNetwork if the network is using a linear layer for the first hidden network.
     * @param learningRatesToTry
     * @param maxEpochsToTry
     * @param patienceToTry
     */
    public HyperparameterTuning(boolean linearNetwork, int indims, int hiddimsEmbedding, int hiddimsOthers, int outdims, VocabClassifier vocabClassifier, List<Double> learningRatesToTry, List<Integer> maxEpochsToTry, List<Integer> patienceToTry) {
        this.linearNetwork = linearNetwork;
        this.indims = indims;
        this.hiddimsEmbedding = hiddimsEmbedding;
        this.hiddimsOthers = hiddimsOthers;
        this.outdims = outdims;
        this.vocabClassifier = vocabClassifier;
        this.learningRatesToTry = learningRatesToTry;
        this.maxEpochsToTry = maxEpochsToTry;
        this.patienceToTry = patienceToTry;
    }

    /**
     * Use the randomized search method to find the best hyperparameters for the model.
     * It is very IMPORTANT to only use the validation set to determine hyperparameters -> avoid overfitting.
     *
     * @param iterations
     */
    public void randomizedSearch(int iterations, VocabDataset trainset, VocabDataset devset) {
        // Store the best validation accuracy found so far.
        double bestAccuracy = 0;

        // Store the variable values of the best run so far.
        double learningRateOnBestResult = 0;
        int maxEpochsOnBestResult = 0;
        int patienceOnBestResult = 0;

        System.out.println("HYPERPARAMETER TUNING STARTING");

        // Try out different combinations of hyperparameters.
        for (int i = 0; i < iterations; i++) {
            // Make a new temporary net to train using the randomly selected hyperparameters.
            Sequential net = createNewNetwork();

            // Get random index of value to try for each hyperparameter.
            double randomLearningRate = learningRatesToTry.get(getRandomIndex(learningRatesToTry.size()));
            int randomMaxEpochs = maxEpochsToTry.get(getRandomIndex(maxEpochsToTry.size()));
            int randomPatience = patienceToTry.get(getRandomIndex(patienceToTry.size()));

            System.out.println("\nITERATION: " + i);

            System.out.println("HYPER-PARAMETERS CHOSEN:\nlearningRate: " + randomLearningRate + ", maxEpochs: " + randomMaxEpochs + ", patience: " + randomPatience);
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

        System.out.println("-------------------------------------------------------");
        System.out.println("\nFINISHED TUNING");
        System.out.println("BEST HYPER-PARAMETERS FOUND \n learningRate: " + learningRateOnBestResult + ", maxEpochs: " + maxEpochsOnBestResult + ", patience: " + patienceOnBestResult);
    }

    public int getRandomIndex(int listSize) {
        return (int) (Math.random() * listSize);
    }


    public Sequential createNewNetwork() {
        Sequential newNet;

        if (linearNetwork) {
            newNet = new Sequential(new Layer[]{
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
        } else {
            newNet = new Sequential(new Layer[]{
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
        }


        return newNet;
    }

}
