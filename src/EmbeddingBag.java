package src;

import org.jblas.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

import minet.layer.init.*;
import minet.layer.Layer;

/**
 * A class for Embedding bag layers. Feel free to modify this class for your implementation.
 */
public class EmbeddingBag implements Layer, java.io.Serializable {

    private static final long serialVersionUID = -10445336293457309L;
    DoubleMatrix W;  // weight matrix (for simplicity, we can ignore the bias term b)
    // for backward
    List<int[]> X;  // store input X for computing backward, each element in this list is a sample (an array of word indices).
    DoubleMatrix gW;    // gradient of W
    private int vocabSize, samplesInTheBatch;

    /**
     * Constructor for EmbeddingBag
     *
     * @param vocabSize (int) vocabulary size
     * @param outdims   (int) output of this layer
     * @param wInit     (WeightInit) weight initialisation method
     */
    public EmbeddingBag(int vocabSize, int outdims, WeightInit wInit) {
        this.vocabSize = vocabSize;
        this.W = wInit.generate(vocabSize, outdims);
        this.gW = DoubleMatrix.zeros(vocabSize, outdims);
    }

    /**
     * Forward pass
     *
     * @param input (List<int[]>) input for forward calculation
     * @return a [batchsize x outdims] matrix, each row is the output of a sample in the batch
     */
    @Override
    public DoubleMatrix forward(Object input) {
        // Calculate number of samples in the batch.
        this.samplesInTheBatch = ((DoubleMatrix) input).length / vocabSize;
        DoubleMatrix Y = null; // output of this layer (to be computed by you)
        List<int[]> X = getX(input);

        System.out.println("ROWS: "+W.getRows()+ "COLUMNS: "+W.getColumns());

        for (int i=0; i<samplesInTheBatch; i++) {
            
            double sumOfWeightsForNode = getSumOfWeights(X.get(i), i);

        }


        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // YOUR CODE HERE

        return null; // there is no need to compute gX as the previous layer of this one is the input layer of the network
    }

    @Override
    public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights) {
        weights.add(W);
        return weights;
    }

    @Override
    public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads) {
        grads.add(gW);
        return grads;
    }

    @Override
    public String toString() {
        return String.format("Embedding: %d rows, %d dims", W.rows, W.columns);
    }

    /**
     * Get the X as a list of integer arrays.
     *
     * @param input
     * @return
     */
    private List<int[]> getX(Object input) {
        DoubleMatrix X = (DoubleMatrix) input;

        // Find all the indexes for each sample where words occur.
        List<int[]> xIndexes = new ArrayList<>();
        for (int i = 0; i < samplesInTheBatch; i++) {
            xIndexes.add(getIndexesWhereOne(X.getRow(i).elementsAsList()));
        }
        return xIndexes;
    }

    /**
     * Get all the indexes where there is a value of 1.
     *
     * @param list
     * @return
     */
    private int[] getIndexesWhereOne(List<Double> list) {
        ArrayList<Integer> indexes = new ArrayList<>();

        // get only the indexes where the value is 1.
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == 1) {
                indexes.add(i);
            }
        }
        // map back to array.
        return indexes.stream().mapToInt(i -> i).toArray();
    }

    private double getSumOfWeights(int[] indexes, int sampleNumber) {
        List<Double> weightsForSample = W.getColumn(sampleNumber).elementsAsList();
        System.out.println(W.getColumn(1).elementsAsList().size());

        List<Double> appropriateWeights = new ArrayList<>();
        IntStream.of(indexes).forEach(i -> appropriateWeights.add(weightsForSample.get(i)));

        // return sum of weights as a double.
        return appropriateWeights.stream().mapToDouble(i -> i.doubleValue()).sum();
    }


}
