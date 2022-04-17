package src;

import org.jblas.*;

import java.util.ArrayList;
import java.util.List;

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
    private int vocabSize, batchSize, outdims;

    /**
     * Constructor for EmbeddingBag
     *
     * @param vocabSize (int) vocabulary size
     * @param outdims   (int) output of this layer
     * @param wInit     (WeightInit) weight initialisation method
     */
    public EmbeddingBag(int vocabSize, int outdims, WeightInit wInit) {
        this.outdims = outdims;
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
        this.batchSize = ((DoubleMatrix) input).length / vocabSize;

        DoubleMatrix Y = new DoubleMatrix(batchSize, W.getColumns()); // output of this layer
        X = getX(input);

        // Iterate through the samples in the batch.
        for (int i = 0; i < batchSize; i++) {
            // iterate through the out dimensions.
            for (int d = 0; d < outdims; d++) {
                double sumOfWeightsForNode = getSumOfWeights(X.get(i), d); // get sum of weights for the node.
                Y.put(i, d, sumOfWeightsForNode);
            }
        }

        return Y;
    }

    // TODO: matrix multiplication with matmul.
    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // Iterate through the out dimensions / nodes.
        for (int d = 0; d < outdims; d++) {
            // Iterate through the samples in the batch.
            for (int s = 0; s < batchSize; s++) {
                int[] indexes = X.get(s); // get indexes of current sample.

                // update gW at the specific index and dimension - with the value calculated.
                for (int i = 0; i < indexes.length; i++) {
                    // calculate value for current sample.
                    double val = gY.get(s, d);
                    // get the prior (before updating - summing) gradient value of the current index and the dimension.
                    double prior = gW.get(indexes[i], d);
                    gW.put(indexes[i], d, prior + val);
                }
            }
        }

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
        for (int i = 0; i < batchSize; i++) {
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

    private double getSumOfWeights(int[] indexes, int dimensionNumber) {
        double sumWeights = 0;

        for (int i = 0; i < indexes.length; i++) {
            sumWeights += W.get(indexes[i], dimensionNumber);
        }

        // return sum of weights as a double.
        return sumWeights;
    }


}
