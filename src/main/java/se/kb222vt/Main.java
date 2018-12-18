package se.kb222vt;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Arrays;
import java.util.Random;

/*
 * Handbook for Weka API: http://prdownloads.sourceforge.net/weka/WekaManual-3-8-3.pdf?download
 */
public class Main {
	
	public static void main(String[] attrs) throws Exception {
		System.out.println("Im alive!");
		//new weka.gui.Main().main(new String[] {}); //start explorer after this
		Instances spiral = DataSource.read("src/main/resources/data/spiral/spiral.arff");
		spiral.setClassIndex(2); //There's, 3 attributes and the class attribute is defined last
		runLinearClassifier(spiral);
		runNeuralNetworkClassifier(spiral);

	}
	
	/**
	 * Run WEKA linear classifier Logistic on data set
	 * Will evaluate the classifier and print the result 
	 * Evaluates with cross validation since data is same for training and learning
	 * @param data data set to run classifier on
	 */
	private static void runLinearClassifier(Instances data) throws Exception {
		System.out.println("##### Run logistic classifier #####");
		Logistic logisticClassifier = new Logistic();
		System.out.println("Run logisticClassifier with options: " + Arrays.toString(logisticClassifier.getOptions()));
		/*
		 * WE SHOULDNT BUILD THE CLASSIFIER, EVALUATION WILL TAKE CARE OF THIS FOR US WHEN USING CROSS VALIDATION
		 * logisticClassifier.buildClassifier(data);
		 * https://waikato.github.io/weka-wiki/use_weka_in_your_java_code/#evaluating
		 */
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(logisticClassifier, data, 10, new Random());
		System.out.println(eval.toSummaryString("\nResults\n\n", false));
		System.out.println(eval.toMatrixString());
		System.out.println("##### End logistic classifier #####");
	}
	
	/**
	 * Run WEKA neural network classifier MultilayerPerceptron on data set
	 * Will evaluate the classifier and print the result
	 * Evaluates with cross validation since data is same for training and learning
	 * @param data data set to run classifier on
	 * @throws Exception 
	 */
	private static void runNeuralNetworkClassifier(Instances data) throws Exception {
		System.out.println("##### Run MultilayerPerceptron classifier #####");
		MultilayerPerceptron multilayerPerceptronClassifier = new MultilayerPerceptron();
		System.out.println("Run multilayerPerceptronClassifier with options: " + Arrays.toString(multilayerPerceptronClassifier.getOptions()));
		multilayerPerceptronClassifier.setHiddenLayers("72");
		//String[] options = Utils.splitOptions("-H 72"); multilayerPerceptronClassifier.setOptions(options); //same as above
		System.out.println("Layers used: " + multilayerPerceptronClassifier.getHiddenLayers());
		/*
		 * WE SHOULDNT BUILD THE CLASSIFIER, EVALUATION WILL TAKE CARE OF THIS FOR US WHEN USING CROSS VALIDATION
		 * multilayerPerceptronClassifier.buildClassifier(data);
		 * https://waikato.github.io/weka-wiki/use_weka_in_your_java_code/#evaluating
		 */
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(multilayerPerceptronClassifier, data, 10, new Random());
		System.out.println(eval.toSummaryString("\nResults\n\n", false));
		System.out.println(eval.toMatrixString());
		System.out.println("##### Run MultilayerPerceptron classifier #####");
	}
	
    public boolean returnTrue() {
        return true;
    }
}
