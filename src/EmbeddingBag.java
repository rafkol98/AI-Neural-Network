package src;

import org.jblas.*;
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

    /**
     * Constructor for EmbeddingBag
     * @param vocabSize (int) vocabulary size
     * @param outdims (int) output of this layer
     * @param wInit (WeightInit) weight initialisation method
     */
    public EmbeddingBag(int vocabSize, int outdims, WeightInit wInit) {
        // YOUR CODE HERE
    }

    /**
     * Forward pass
     * @param input (List<int[]>) input for forward calculation
     * @return a [batchsize x outdims] matrix, each row is the output of a sample in the batch
     */
    @Override
    public DoubleMatrix forward(Object input) {
        DoubleMatrix Y = null; // output of this layer (to be computed by you)

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

}
