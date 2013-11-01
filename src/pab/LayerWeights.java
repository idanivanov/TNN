package pab;

import java.util.Random;
import java.util.Vector;

public class LayerWeights {
	
	private Vector<Vector<Double>> weights;
	
	public LayerWeights(Vector<Vector<Double>> weights) {
		this.setWeights(weights);
	}
	
	public LayerWeights(int inputLayerNeuronsCount, int outputLayerNeuronsCount) {
		Random generator = new Random(0);
		
		++inputLayerNeuronsCount; // For the BIAS
		
		weights = new Vector<Vector<Double>>(inputLayerNeuronsCount);
		
		for (int i = 0; i < inputLayerNeuronsCount; i++) {
			weights.add(new Vector<Double>(outputLayerNeuronsCount));
			for (int k = 0; k < outputLayerNeuronsCount; k++) {
				generator.setSeed(-(k * 1000) + (i * 1000)); // FIXME II: not sure if this is a good seed
				weights.elementAt(i).add(4. * generator.nextDouble() - 2.);
			}
		}
	}
	
	public void setWeights(Vector<Vector<Double>> weights) {
		this.weights = weights;
	}
	
	public double get(int i, int o) {
		return weights.elementAt(i).elementAt(o);
	}
	
	public void change(int i, int o, double difference) {
		Double w = weights.elementAt(i).elementAt(o);
		w += difference;
		weights.elementAt(i).setElementAt(w, o);
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		Vector<Double> neuronWeights;
		
		for (int i = 0, ii = this.weights.size(); i < ii; i++) {
			neuronWeights = this.weights.elementAt(i);
			for (int o = 0, oo = neuronWeights.size(); o < oo; o++) {
				sb.append("(i=" + i + ",o=" + (o + 1) + ")=" + neuronWeights.elementAt(o) + "\t");
			}
			sb.append("\n");
		}
		
		return sb.toString();
	}
	
}
