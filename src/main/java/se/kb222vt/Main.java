package se.kb222vt;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Arrays;

/*
 * Handbook for Weka API: http://prdownloads.sourceforge.net/weka/WekaManual-3-8-3.pdf?download
 */
public class Main {
	
	public static void main(String[] attrs) throws Exception {
		System.out.println("Im alive!");
		//new weka.gui.Main().main(new String[] {}); //start explorer after this
		Instances spiral = DataSource.read("src/main/resources/data/spiral/spiral.arff");
		spiral.setClassIndex(2); //There's, 3 attributes and the class attribute is defined last

		System.out.println("##### Run logistic classifier #####");
		Logistic logisticClassifier = new Logistic();
		System.out.println("Run logisticClassifier with options: " + Arrays.toString(logisticClassifier.getOptions()));
		logisticClassifier.buildClassifier(spiral);
		Evaluation eval = new Evaluation(spiral);
		eval.evaluateModel(logisticClassifier, spiral);
		System.out.println(eval.toSummaryString("\nResults\n\n", false));
		System.out.println(eval.toMatrixString());
		System.out.println("##### End logistic classifier #####");

		System.out.println("##### Run MultilayerPerceptron classifier #####");
		MultilayerPerceptron multilayerPerceptronClassifier = new MultilayerPerceptron();
		System.out.println("Run multilayerPerceptronClassifier with options: " + Arrays.toString(multilayerPerceptronClassifier.getOptions()));
		multilayerPerceptronClassifier.setHiddenLayers("72");
		//String[] options = Utils.splitOptions("-H 72"); multilayerPerceptronClassifier.setOptions(options); //same as above
		System.out.println("Layers used: " + multilayerPerceptronClassifier.getHiddenLayers());
		multilayerPerceptronClassifier.buildClassifier(spiral);
		Evaluation eval2 = new Evaluation(spiral);
		eval2.evaluateModel(multilayerPerceptronClassifier, spiral);
		System.out.println(eval2.toSummaryString("\nResults\n\n", false));
		System.out.println(eval2.toMatrixString());
		System.out.println("##### Run MultilayerPerceptron classifier #####");

	}
	
    public boolean returnTrue() {
        return true;
    }
}
