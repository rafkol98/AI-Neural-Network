package src;

import minet.data.Dataset;
import minet.util.Pair;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class VocabDataset  extends Dataset<double[], Integer> {

    // number of input features
    int inputDims;

    /**
     * Constructor for VocabDataset.
     *
     * @param batchsize (int) size of each mini-batch
     * @param shuffle   (boolean) if true, shuffle the dataset at the beginning of each epoch
     * @param rnd       (java.util.Random) random generator for the shuffling
     */
    public VocabDataset(int batchsize, boolean shuffle, Random rnd) {
        super(batchsize, shuffle, rnd);
    }

    @Override
    public void fromFile(String path) throws IOException {
    }


    /**
     * Load MNIST data from file and vocabulary.
     */
    public void fromFile(String path, String pathVocabulary) throws IOException {
        items = new ArrayList<Pair<double[], Integer>>();

        System.out.println("here"+items.size());



        // the number of features and dimensions are equal.
        //TODO: improve!
        int instances = countLinesInFile(path);
        int noFeatures = countLinesInFile(pathVocabulary);

        System.out.println("instances: "+instances + " , features: "+ noFeatures);
        FileReader fr = new FileReader(path);
        BufferedReader br = new BufferedReader(fr);

        for (int i = 0; i < 5; i++) {
            String[] line = br.readLine().split(" ; ");
            String locations = line[0]; // get all the locations (indexes) of the words.
            int y = Integer.valueOf(line[1]);
            System.out.println(locations + " y:"+y);
            ArrayList<Integer> encoded = oneHotEncode(noFeatures, locations);

            System.out.println(encoded.toString());

        }

        br.close();
    }


    public int countLinesInFile(String path) throws IOException {
        BufferedReader br = new BufferedReader( new FileReader(path));
        int lines = 0;
        while (br.readLine() != null) lines++;
        br.close();
        return lines;
    }

    public ArrayList<Integer> oneHotEncode(int noFeatures, String locations) {
        // initialise with noFeatures of zeros.
        ArrayList<Integer> encoded = new ArrayList<Integer>(Collections.nCopies(noFeatures, 0));
        // Get all the indexes of the items in a list.
        List<Integer> indexes = Arrays.stream(locations.split(" "))
                .map(Integer::parseInt)
                .collect(Collectors.toList());

        // update zeros with 1 at all the indexes of elements.
        for (int i=0; i<encoded.size(); i++) {
            if (indexes.contains(i)) {
                encoded.add(i, 1);
            }
        }
        return encoded;
    }
}
