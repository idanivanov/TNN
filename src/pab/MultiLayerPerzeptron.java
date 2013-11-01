package pab;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Vector;

public class MultiLayerPerzeptron {
	
	public static int MAX_NEURONS = 1000;
	public static int MAX_LAYERS = 4;
	public static double TRAINING_RATE = .35;
	
	private Vector<Pattern> patterns;
	private Vector<LayerWeights> weights;
	private Integer[] structure;
	private Vector<Double> expectedOutputs; // this is used only for the current pattern while teaching
	
	// Example: structure = new Integer[]{4, 10, 2};
	public MultiLayerPerzeptron(Integer[] structure) throws Exception {
		ArrayList<Integer> temp = new ArrayList<Integer>(Arrays.asList(structure));
		
		if (structure.length < 2 ||
			structure.length > MAX_LAYERS ||
			Collections.max(temp) > MAX_NEURONS ||
			Collections.min(temp) < 1) {
			throw new Exception("Illegal structure!");
		}
		this.patterns = new Vector<Pattern>();
		this.structure = structure;
		this.weights = new Vector<LayerWeights>(structure.length - 1);
		for (int i = 0, ii = structure.length - 1; i < ii; i++) {
			this.weights.add(new LayerWeights(structure[i], structure[i + 1]));
		}
	}
	
	public void readPatterns(String fileName) throws IOException {
		Pattern pattern;
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String line;
		String[] lineXY, buff;
		int i;
		
		try {
			// skip headers
			br.readLine();
			br.readLine();
			
			// read pattern lines
			while ((line = br.readLine()) != null) {
				pattern = new Pattern();
				lineXY = line.split("    ");
				// get X
				buff = lineXY[0].split(" ");
				for (i = 0; i < buff.length; i++) {
					if (!buff[i].isEmpty()) {
						pattern.x.add(Double.valueOf(buff[i]));
					}
				}
				// get Y
				buff = lineXY[1].split(" ");
				for (i = 0; i < buff.length; i++) {
					if (!buff[i].isEmpty()) {
						pattern.y.add(Double.valueOf(buff[i]));
					}
				}
				this.patterns.add(pattern);
			}
		}
		finally {
			br.close();
		}
	}
	
	private double transferFunction(double sum, int type) {
		return 1. / (1. + Math.exp(-sum)); // TODO
	}
	
	private double transferFunctionDerivative(double output, int type) {
		return output * (1. - output); // TODO
	}
	
	private double calculateNeuronOutput(Vector<Double> inputs, int layerIndex, int neuronIndex, int transferFunction) {
		LayerWeights layerWeights = this.weights.elementAt(layerIndex - 1);
		double sum = layerWeights.get(0, neuronIndex); // start with BIAS-weight
		
		for (int i = 1, ii = inputs.size(); i <= ii; i++) {
			sum += layerWeights.get(i, neuronIndex) * inputs.elementAt(i - 1);
		}
		return transferFunction(sum, transferFunction);
	}
	
	private void changeWeights(Vector<Double> inputs, Vector<Double> deltas, int layerIndex) {
		LayerWeights layerWeights = this.weights.elementAt(layerIndex);
		double difference;
		
		for (int i = 0; i < inputs.size(); i++) {
			for (int o = 0; o < deltas.size(); o++) {
				difference = MultiLayerPerzeptron.TRAINING_RATE * deltas.elementAt(o) * inputs.elementAt(i);
				layerWeights.change(i, o, difference);
			}
		}
	}
	
	// calculate deltas for the hidden layers
	private Vector<Double> calculateDeltas(Vector<Double> outputs, int layerIndex, Vector<Double> nextLayerDeltas, int transferFunction) {
		Vector<Double> deltas = new Vector<Double>(outputs.size());
		LayerWeights nextLayerWeights = this.weights.elementAt(layerIndex);
		double nextLayerDeltaWeightedSum, delta;
		
		for (int i = 0, ii = outputs.size(); i < ii; i++) {
			nextLayerDeltaWeightedSum = 0;
			for (int k = 1, kk = nextLayerDeltas.size(); k < kk; k++) {
				nextLayerDeltaWeightedSum += nextLayerWeights.get(i + 1, k) * nextLayerDeltas.elementAt(k);
			}
			delta = nextLayerDeltaWeightedSum * 
					this.transferFunctionDerivative(outputs.elementAt(i), transferFunction);
			deltas.add(delta);
		}
		
		return deltas;
	}
	
	// calculate deltas for the output layer
	private Vector<Double> calculateDeltas(Vector<Double> outputs, int transferFunction) {
		Vector<Double> deltas = new Vector<Double>(outputs.size());
		double delta;
		
		for (int i = 0, ii = outputs.size(); i < ii; i++) {
			delta = (this.expectedOutputs.elementAt(i) - outputs.elementAt(i)) * 
					this.transferFunctionDerivative(outputs.elementAt(i), transferFunction);
			deltas.add(delta);
		}
		
		return deltas;
	}
	
	// recursive function, returns vector of delta
	private Vector<Double> training(Vector<Double> inputs, int layerIndex) {
		int neuronsInLayer = this.structure[layerIndex];
		Vector<Double> outputs = new Vector<Double>(neuronsInLayer);
		Vector<Double> deltas, nextLayerDeltas;
		int transferFunction = 1; // TODO
		
		// calculate outputs of the neurons in the layer
		for (int i = 0; i < neuronsInLayer; i++) {
			outputs.add(calculateNeuronOutput(inputs, layerIndex, i, transferFunction));
		}
		
		// check if hidden layer
		if (layerIndex < this.structure.length - 1) {
			// recursively call the function to get deltas of the next layer
			nextLayerDeltas = training(outputs, layerIndex + 1);
			// calculate deltas of the current layer
			deltas = calculateDeltas(outputs, layerIndex, nextLayerDeltas, transferFunction);
			// change the weights between this layer and the next one
			changeWeights(outputs, nextLayerDeltas, layerIndex);
		}
		else {
			// if it's the output layer just calculate delta
			deltas = calculateDeltas(outputs, transferFunction);
		}
		
		return deltas;
	}
	
	public void teachByPatterns() {
		Vector<Double> deltas;
		
		for (Pattern pattern : this.patterns) {
			this.expectedOutputs = pattern.y;
			deltas = training(pattern.x, 1);
			// change the weights between the input and the first hidden layer
			changeWeights(pattern.x, deltas, 0);
		}
	}
	
	public void printPatterns() {
		Pattern pattern;
		int i, k;
		
		for (i = 0; i < this.patterns.size(); i++) {
			pattern = this.patterns.elementAt(i);
			System.out.print("Pattern " + (i + 1) + ": X{");
			for (k = 0; k < pattern.x.size() - 1; k++) {
				System.out.print(pattern.x.elementAt(k) + ", ");
			}
			System.out.print(pattern.x.elementAt(k) + "}; Y(");
			for (k = 0; k < pattern.y.size() - 1; k++) {
				System.out.print(pattern.y.elementAt(k) + ", ");
			}
			System.out.print(pattern.y.elementAt(k) + "};\n");
		}
	}
	
	public void printWeights() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0, ii = this.weights.size(); i < ii; i++) {
			sb
				.append("\nWeights from layer " + (i + 1) + " to " + (i + 2) + ":\n")
				.append(this.weights.elementAt(i).toString())
				.append("\n");
		}
		System.out.print(sb.toString());
	}
	
	public static void main(String[] args) throws Exception {
		MultiLayerPerzeptron mlp = new MultiLayerPerzeptron(new Integer[]{4, 10, 10, 2});
		mlp.readPatterns("data/training.dat");
		mlp.printPatterns();
		mlp.printWeights();
		mlp.teachByPatterns();
		mlp.printWeights();
	}

}
