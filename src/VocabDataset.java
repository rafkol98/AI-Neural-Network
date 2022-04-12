package src;

import minet.data.Dataset;
import minet.util.Pair;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

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

    /**
     * Load MNIST data from file.
     */
    @Override
    public void fromFile(String path) throws IOException {
        items = new ArrayList<Pair<double[], Integer>>();

        //TODO: remove this
        File myFile = new File(path);
        System.out.println("Attempting to read from file in: "+myFile.getCanonicalPath());

        BufferedReader br = new BufferedReader(new FileReader(path));

        // first line
        String[] ss = br.readLine().split(" ");
        int size = Integer.valueOf(ss[0]);
        inputDims = Integer.valueOf(ss[1]);

        for (int i = 0; i < size; i++) {
            ss = br.readLine().split(" ; ");
            String[] sx = ss[0].split(" ");
            double[] xs = new double[inputDims];
            Integer y = Integer.valueOf(ss[1]);
            for (int j = 0; j < sx.length; j++) {
                xs[j] = Double.parseDouble(sx[j]);
            }
            items.add(new Pair<double[], Integer>(xs, y));
        }
        br.close();
    }
}
