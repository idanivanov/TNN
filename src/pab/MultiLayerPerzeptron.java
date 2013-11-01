package pab;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Vector;

public class MultiLayerPerzeptron {
	
	public static final int MAX_NEURONS = 1000;
	public static final int MAX_LAYERS = 4;
	public static final double LEARNING_RATE = .35;
	
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
	
	private byte[] toByteArray(double value) {
	    byte[] bytes = new byte[8];
	    ByteBuffer.wrap(bytes).putDouble(value);
	    return bytes;
	}

	private double toDouble(byte[] bytes) {
	    return ByteBuffer.wrap(bytes).getDouble();
	}
	
	public void writeWeights(String filePath) throws IOException {
		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(filePath));
		LayerWeights layerWeights;
		Integer inputsCount;
		
		try {
			for (int layer = 0; layer < this.weights.size(); layer++) {
				layerWeights = this.weights.elementAt(layer);
				inputsCount = this.structure[layer] + 1; // count of inputs + BIAS
				bos.write(inputsCount.byteValue()); // write count of inputs + BIAS
				bos.write(this.structure[layer + 1].byteValue()); // write count of outputs
				for (int i = 0; i < inputsCount; i++) {
					for (int o = 0, oo = this.structure[layer + 1]; o < oo; o++) {
						bos.write(toByteArray(layerWeights.get(i, o))); // write weights
					}
				}
			}
		}
		finally {
			bos.close();
		}
	}
	
	public void readWeights(String filePath) throws IOException {
		BufferedInputStream bis = new BufferedInputStream(new FileInputStream(filePath));
		Vector<Vector<Double>> layerWeightsVector;
		Vector<Double> neuronWeights;
		int inputsCount, outputsCount;
		byte[] buffer = new byte[8];
		
		this.weights = new Vector<LayerWeights>();
		
		try {
			while((inputsCount = bis.read()) != -1) {
				outputsCount = bis.read();
				layerWeightsVector = new Vector<Vector<Double>>(inputsCount);
				for (int i = 0; i < inputsCount; i++) {
					neuronWeights = new Vector<Double>(outputsCount);
					for (int o = 0; o < outputsCount; o++) {
						bis.read(buffer);
						neuronWeights.add(toDouble(buffer));
					}
					layerWeightsVector.add(neuronWeights);
				}
				this.weights.add(new LayerWeights(layerWeightsVector));
			}
		}
		finally {
			bis.close();
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
				difference = MultiLayerPerzeptron.LEARNING_RATE * deltas.elementAt(o) * inputs.elementAt(i);
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
		//mlp.printWeights();
		mlp.teachByPatterns();
		mlp.printWeights();
		mlp.writeWeights("data/weights.tnn");
		mlp.readWeights("data/weights.tnn");
		mlp.printWeights();
	}

}
