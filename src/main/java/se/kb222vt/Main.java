package se.kb222vt;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Random;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

/*
 * Handbook for Weka API: http://prdownloads.sourceforge.net/weka/WekaManual-3-8-3.pdf?download
 * ScatterChart with javaFX: https://docs.oracle.com/javafx/2/charts/scatter-chart.htm
 */
public class Main extends Application {
	
	@Override
	public void start(Stage stage) throws Exception {
		//new weka.gui.Main().main(new String[] {}); //start explorer after this
		Instances spiral = DataSource.read("src/main/resources/data/spiral/spiral.arff");
		spiral.setClassIndex(2); //There's, 3 attributes and the class attribute is defined last
		
		runLinearClassifier(spiral);
		runNeuralNetworkClassifier(spiral);
		HashMap<String, XYChart.Series<Double, Double>> series = getSeriesFromData(spiral);
		ScatterChart scatterChart = new ScatterChart(new NumberAxis(), new NumberAxis());
		for(XYChart.Series<Double, Double> serie : series.values()) {
			scatterChart.getData().add(serie);
		}
        Scene scene  = new Scene(scatterChart, 500, 500);
        scene.getStylesheets().add("style.css");
        stage.setScene(scene);
        stage.show();
		
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
		System.out.println(eval.toSummaryString("Summary", false));
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
		System.out.println(eval.toSummaryString("Summary", false));
		System.out.println(eval.toMatrixString());
		System.out.println("##### End MultilayerPerceptron classifier #####");
	}
	
	/**
	 * Get series from the Instances data, expected format of data is {X-cord, Y-cord, Type}
	 * @param data The data to create a scatter chart from, expected format {Double,Double,String}
	 */
    public static HashMap<String, XYChart.Series<Double, Double>> getSeriesFromData(Instances data) {
    	HashMap<String, XYChart.Series<Double, Double>> series = new HashMap<>(); //<Type, Serie of points>
    	for(Instance inst : data) {
    		String[] pointData = inst.toString().split(",");
			double x = Double.parseDouble(pointData[0]);
			double y = Double.parseDouble(pointData[1]);
    		String type = pointData[2];
    		XYChart.Data<Double, Double> point = new XYChart.Data<>(x, y);
    		if(series.containsKey(type)) {
    			series.get(type).getData().add(point);
    		}else {
    			//this series isnt in the result hashMap yet, add it
    			XYChart.Series<Double, Double> serie = new XYChart.Series<Double, Double>();
    			serie.setName(type);
    			serie.getData().add(point);
    			series.put(type, serie);
    		}
    	}
		return series;
    }
}
