package src;

import org.jblas.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import minet.layer.init.*;
import minet.layer.Layer;

/**
 * A class for Embedding bag layers. Feel free to modify this class for your implementation.
 */
public class EmbeddingBag implements Layer, java.io.Serializable {			

	private static final long serialVersionUID = -10445336293457309L;

	private int vocabSize;
	DoubleMatrix W;  // weight matrix (for simplicity, we can ignore the bias term b)

    // for backward
    List<int[]> X;  // store input X for computing backward, each element in this list is a sample (an array of word indices).
    DoubleMatrix gW;    // gradient of W

    /**
     * Constructor for EmbeddingBag
     * @param vocabSize (int) vocabulary size
     * @param outdims (int) output of this layer
     * @param wInit (WeightInit) weight initialisation method
     */
    public EmbeddingBag(int vocabSize, int outdims, WeightInit wInit) {
        this.vocabSize = vocabSize;
        this.W = wInit.generate(vocabSize, outdims);
        this.gW = DoubleMatrix.zeros(vocabSize, outdims);
    }

    /**
     * Forward pass
     * @param input (List<int[]>) input for forward calculation
     * @return a [batchsize x outdims] matrix, each row is the output of a sample in the batch
     */
    @Override
    public DoubleMatrix forward(Object input) {
        DoubleMatrix Y = null; // output of this layer (to be computed by you)
        getX(input);

        // YOUR CODE HERE


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

    public List<int[]> getX(Object input) {
        DoubleMatrix X = (DoubleMatrix)input;
        //calculate batch size
        int batchSize = ((DoubleMatrix) input).length / vocabSize;

        List<int[]> xIndexes = new ArrayList<>();
        int indexStart = 0;
        int indexEnd = vocabSize;
        for (int i=0; i<batchSize; i++) {
            xIndexes.add(getIndexesWhereOne(X.getRow(i).elementsAsList()));
            
            if (i == 2) {
                System.out.println("\n\n DEBUG X embedding: "+X.getRow(2));
                System.out.println("\n\nDEBUG EMBEDDING:"+ Arrays.toString(getIndexesWhereOne(X.getRow(i).elementsAsList())));
            }
        }
        return null;
    }

    /**
     * Get all the indexes where there is a value of 1.
     * @param list
     * @return
     */
    private int[] getIndexesWhereOne(List<Double> list) {
        ArrayList<Integer> indexes = new ArrayList<>();

        for (int i=0; i<list.size();i++) {
            if (list.get(i) == 1) {
                indexes.add(i);
            }
        }

        return indexes.stream().mapToInt(i -> i).toArray();
    }


}
