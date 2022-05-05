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
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import src.EmbeddingBag;

public class A4Main {

    // variables used for hyperparameter tuning.
    private static List<Double> learningRatesToTry;
    private static List<Integer> maxEpochsToTry = new ArrayList<>();
    private static List<Integer> patienceToTry = new ArrayList<>();
    private static int iterations = 1;

    /**
     * Example A4Main class. Feel free to edit this file
     */
    public static void main(String[] args) throws IOException {

        if (args.length < 6) {
            System.out.println("Usage: java A4Main <part1/part2/part3/part4/part5> <seed> <trainFile> <devFile> <testFile> <vocabFile> <classesFile>");
            return;
        }

        boolean useTrainedWeights = false;
        boolean freeze = false;

        if (args[0].equalsIgnoreCase("part3") || args[0].equalsIgnoreCase("part4") || args[0].equalsIgnoreCase("part5")) {
            useTrainedWeights = true;
            // if part 4 or 5, then freeze the weights - i.e. do not update them during training.
            freeze = args[0].equalsIgnoreCase("part3") ? false : true;
        }

        // set jblas random seed (for reproducibility)
        int seed = Integer.parseInt(args[1]);
        org.jblas.util.Random.seed(seed);
        Random rnd = new Random(seed);

        // turn off jblas info messages
        Logger.getLogger().setLevel(Logger.WARNING);

        int batchsize = 50;
        int hiddimsEmbedding, hiddimsOthers;

        // If we are using the words2vec embedding then the hidden dimensions of both embedding and the following
        // layers are increased to 300 and 400 respectively.
        if (args[0].equalsIgnoreCase("part5")) {
            hiddimsEmbedding = 300;
            hiddimsOthers = 400;
        }
        // Otherwise they have size of 100 and 200.
        else {
            hiddimsEmbedding = 100;
            hiddimsOthers = 200;
        }

        // Hyperparameters.
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

        boolean verbose = true;
        boolean tune = false;
        if (args.length >= 8) {
            // Silent mode - not printing each step
            if (args[7].equalsIgnoreCase("silent")) {
                verbose = false;
            }

            // Whether to run the randomized search hyperparameter tuner.
            if (args.length >= 9 && args[8].equalsIgnoreCase("tune")) {
                tune = true;
            }
        }


        // Arraylists containing values to try in Hyperparameter tuning.
        if (tune) {
            getScannerInputValues();
        }

        VocabClassifier vocabClassifier = new VocabClassifier(verbose);

        // TODO: redundant - make it all the same.
        CrossEntropy loss = new CrossEntropy();
        switch (args[0]) {
            case "part1":
                if (tune) {
                    performHyperparameterTuning(true, trainset, devset, indims, hiddimsEmbedding, hiddimsOthers, outdims, vocabClassifier, learningRatesToTry, maxEpochsToTry, patienceToTry, iterations);
                } else {
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
                }
                break;

            case "part2":
                if (tune) {
                    performHyperparameterTuning(false, trainset, devset, indims, hiddimsEmbedding, hiddimsOthers, outdims, vocabClassifier, learningRatesToTry, maxEpochsToTry, patienceToTry, iterations);
                } else {
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

                }

                break;

            case "part3":
            case "part4":
            case "part5":
                if (tune) {
                    performHyperparameterTuning(false, trainset, devset, indims, hiddimsEmbedding, hiddimsOthers, outdims, vocabClassifier, learningRatesToTry, maxEpochsToTry, patienceToTry, iterations);
                } else {
                    net = new Sequential(new Layer[]{
                            // Input to first hidden layer (Embedding bag). Use pretrained weights.
                            // Decide whether to freeze them or not using the freeze flag.
                            new EmbeddingBag(indims, hiddimsEmbedding, pretrainedWeights, freeze),
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
                }
                break;

            default:
                System.out.println("Please select part1, part2, part3, part4 or part 5.");
        }
    }

    //TODO: catch exceptions.
    public static void getScannerInputValues() {
        System.out.println("-----------------------HYPERPARAMETER TUNING INPUT-----------------------");
        Scanner scanner = new Scanner(System.in);
        System.out.println("\nEnter LEARNING_RATE values to try (DOUBLE values) - Separate them with a space");
        String learningRatesString = scanner.nextLine();
        try {
        // Convert string input into an arralyist of doubles.
        learningRatesToTry = Stream.of(learningRatesString).map(s -> s.split(" "))
                .flatMap((Function<String[], Stream<Double>>) strings -> Stream.of(strings).map(Double::parseDouble))
                .collect(Collectors.toList());

        System.out.println("LEARNING RATES TO TRY: " + learningRatesToTry);

        System.out.println("\nEnter MAX_EPOCHS to try (INTEGER values) - Separate them with a space");
        String maxEpochsString = scanner.nextLine();

        // Convert string input into an arralyist of doubles.
        maxEpochsToTry = Stream.of(maxEpochsString).map(s -> s.split(" "))
                .flatMap((Function<String[], Stream<Integer>>) strings -> Stream.of(strings).map(Integer::parseInt))
                .collect(Collectors.toList());

        System.out.println("MAX EPOCHS TO TRY: " + maxEpochsToTry);

        System.out.println("\nEnter PATIENCE values to try (INTEGER values) - Separate them with a space");
        String patienceString = scanner.nextLine();

        // Convert string input into an arralyist of doubles.
        patienceToTry = Stream.of(patienceString).map(s -> s.split(" "))
                .flatMap((Function<String[], Stream<Integer>>) strings -> Stream.of(strings).map(Integer::parseInt))
                .collect(Collectors.toList());

        System.out.println("PATIENCE TO TRY: " + patienceToTry);

        System.out.println("\nEnter NUMBER OF ITERATIONS (INTEGER value)");
        iterations = scanner.nextInt();

        System.out.println("-------------------------------------------------------");
        } catch (Exception e) {
            System.out.println("\nSYSTEM EXITED: Please follow the instructions on the number format (Integer or Double) for each hyperparameter.");
            System.exit(1);
        }
    }

    /**
     * Perform hyperparamer tuning - Extension.
     */
    public static void performHyperparameterTuning(boolean linearNetwork, VocabDataset trainset, VocabDataset devset, int indims, int hiddimsEmbedding, int hiddimsOthers, int outdims, VocabClassifier vocabClassifier, List<Double> learningRatesToTry, List<Integer> maxEpochsToTry, List<Integer> patienceToTry, int iterations) {
        // perform hyperparameter tuning using randomized search method.
        HyperparameterTuning hyperparameterTuning = new HyperparameterTuning(linearNetwork, indims, hiddimsEmbedding, hiddimsOthers, outdims, vocabClassifier, learningRatesToTry, maxEpochsToTry, patienceToTry);
        hyperparameterTuning.randomizedSearch(iterations, trainset, devset);
    }
}