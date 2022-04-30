package src;

import minet.data.Dataset;
import minet.util.Pair;
import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class VocabDataset extends Dataset<double[], Integer> {

    // number of input features
    private int inputDims;
    private String pathVocabulary;
    private boolean trainingWeights;

    private ArrayList<double[]> allWeights = new ArrayList<>();

    private DoubleMatrix pretrainedWeights;

    public DoubleMatrix getPretrainedWeights() {
        return pretrainedWeights;
    }

    /**
     * Get the number of input dimensions.
     * @return the number of input dimensions.
     */
    public int getInputDims() {
        return inputDims;
    }

    /**
     * Constructor for VocabDataset.
     *
     * @param batchsize (int) size of each mini-batch
     * @param shuffle   (boolean) if true, shuffle the dataset at the beginning of each epoch
     * @param rnd       (java.util.Random) random generator for the shuffling
     */
    public VocabDataset(int batchsize, boolean shuffle, Random rnd, String pathVocabulary, boolean trainingWeights) {
        super(batchsize, shuffle, rnd);
        this.pathVocabulary = pathVocabulary;
        this.trainingWeights = trainingWeights;
    }

    /**
     * Load data from file and vocabulary.
     */
    @Override
    public void fromFile(String path) throws IOException {
        items = new ArrayList<Pair<double[], Integer>>();

        // get the number of instances (elements) and number of features.
        int instances = countLinesInFile(path, trainingWeights, false);
        inputDims = countLinesInFile(pathVocabulary, trainingWeights, true);


        FileReader fr = new FileReader(path);
        BufferedReader br = new BufferedReader(fr);

        // iterate through all the instances
        for (int i = 0; i < instances; i++) {
            String[] line = br.readLine().split(" ; ");
            String locations = line[0]; // get all the locations (indexes) of the words.

            // get the one-hot encoded values and the y value for the current instance.
            double[] encoded = oneHotEncode(inputDims, locations);
            int y = Integer.valueOf(line[1]);

            items.add(new Pair<double[], Integer>(encoded, y));
        }
        br.close();
    }


    /**
     * Count the lines in a given file.
     * @param path the path of the file.
     * @return the number of lines.
     * @throws IOException
     */
    public int countLinesInFile(String path, boolean trainingWeights, boolean vocabulary) throws IOException {
        BufferedReader br = new BufferedReader( new FileReader(path));
        int lines = 0;
        String line;
        while ((line = br.readLine()) != null) {
            // if the trainingWeights flag is true - then read in the weights provided in the file.
            if (trainingWeights && vocabulary) {
                String[] splitLine = line.split(" ", 2);
                placeInWeightslist(splitLine[1]);
            }

            lines++;
        }
        // create double matrix for the weights passed in.
        if (trainingWeights && vocabulary) {
            createDoubleMatrixForWeights();
        }

        br.close();
        return lines;
    }

    /**
     * One hot encode the given instance - Bag-of-Word strategy.
     * @param noFeatures the number of features in the current dataset.
     * @param locations the locations of where we will place a one.
     * @return the one-hot encoded double array.
     */
    public double[] oneHotEncode(int noFeatures, String locations) {
        // initialise with noFeatures of zeros.
        ArrayList<Integer> encoded = new ArrayList<Integer>(Collections.nCopies(noFeatures, 0));
        // Get all the indexes of the items in a list.
        List<Integer> indexes = Arrays.stream(locations.split(" "))
                .map(Integer::parseInt)
                .collect(Collectors.toList());

        // update zeros with 1 at all the indexes of elements.
        for (int i=0; i<encoded.size(); i++) {
            if (indexes.contains(i)) {
                encoded.set(i, 1);
            }
        }

        // Convert to double array and return.
        return encoded.stream().mapToDouble(Integer::doubleValue).toArray();
    }

    /**
     * Convert weight string to a double array containing weight values. This double array is then added
     * to the allWeights ArrayList.
     * @param weight the pre-trained weights of a word as a large string.
     */
    private void placeInWeightslist(String weight) {
        String[] str = weight.split(" ");

        double[] doubleValues = Arrays.stream(str)
                .mapToDouble(Double::parseDouble)
                .toArray();

        allWeights.add(doubleValues);
    }

    private void createDoubleMatrixForWeights() {
        double[][] xs = new double[allWeights.size()][];

        for (int i = 0; i< allWeights.size(); i++) {
            xs[i] = Arrays.stream(allWeights.get(i)).toArray();
        }

        pretrainedWeights = new DoubleMatrix(xs);
    }
}
