package SEE;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Random;

import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.GeneticSearch;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.AbstractEvaluationMetric;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.PluginManager;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.WekaPackageManager;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class SEE1 {
	public static Instances dataSet;
	public static Instances trainSet;
	public static Instances testSet;
	
	private static final String[] searchNames={"Forward","BackWard","Genetic","Full","NSGAII"};
	
	private static final String outDir="E:/MachineLearning/ExperimentData/";
	
	private static final String[] dataSetNames={"maxwell.arff"};
	
	private static final String[] statNames={"FN","MMRE"};
	//Data in.
	private static File frData;
	//Result out.
	private static File fwResult;
	
	private static ASSearch[] searchs=new ASSearch[5];
	
	private static void initialization() throws Exception{
		PluginManager.addPlugin(AbstractEvaluationMetric.class.getName(),"RegressionEval",RegressionEval.class.getName());
		frData=new File(dataSetNames[0]);
		dataSet = new Instances(new FileReader(frData));
		dataSet.setClassIndex(dataSet.numAttributes()-1);
		dataSet.sort(0);
		trainSet = new Instances(dataSet, 0, 50);
        testSet = new Instances(dataSet, 50, 12);
		GreedyStepwise FW=new GreedyStepwise();
		FW.setSearchBackwards(false);
		GreedyStepwise BW=new GreedyStepwise();
		FW.setSearchBackwards(true);
		GeneticSearch GS=new GeneticSearch();
		GS.setMaxGenerations(100);
		NSGAII nsgaii=new NSGAII();
		nsgaii.setMaxGenerations(100);
		nsgaii.setPopulationSize(200);
		searchs[0]=FW;
		searchs[1]=BW;
		searchs[2]=GS;
		searchs[3]=null;
		searchs[4]=nsgaii;
	}
	/*
	private static String expandAttrsName(int a[]){
		String name=String.valueOf(a[0]);
		name+=String.valueOf(a[i]);
	}
	
	private static Instances dimensionExpansion(Instances insts,int dim){
		int numAttrs=insts.numAttributes();
		int numInsts=insts.numInstances();
		int i,j;
		boolean[] attrsIndex=new boolean[numAttrs];
		for(i=0;i<numAttrs;++i)
			if(insts.classIndex()!=i){
				Attribute newAttr=new Attribute(String);
				insts.insertAttributeAt(att, position);
				for(j=0;j<numInsts;++j){
				}
			}
		return insts;
	}*/
	
	private static void randomSplit(Instances dataSet, double d) throws Exception {
		dataSet.randomize(new Random(System.currentTimeMillis()));
		int dataSize = dataSet.numInstances();
		int trainSize = (int)Math.round(dataSize * d);
        int testSize = dataSize - trainSize;
        trainSet = new Instances(dataSet, 0, trainSize);
        testSet = new Instances(dataSet, trainSize, testSize);
	}
	
	private static Instances experiment(ASSearch search) throws Exception {
		int i,j;
		ArrayList<Attribute> attrs=new ArrayList<Attribute>();
		attrs.add(new Attribute("FN"));
		attrs.add(new Attribute("MMRE"));
		Instances results=new Instances("results",attrs,10);
		kNNRegression classifer=new kNNRegression();
		SingleWrapperSubsetEval wrapSE=new SingleWrapperSubsetEval();
		SelectedTag evalMeasure=new SelectedTag("MMRE",WrapperSubsetEval.TAGS_EVALUATION);
		classifer.setK(3);
		wrapSE.setClassifier(classifer);
		wrapSE.setEvaluationMeasure(evalMeasure);
		wrapSE.setFolds(5);
		wrapSE.buildEvaluator(trainSet);
		int[][] temp;
		if (search instanceof NSGAII)
			temp=((NSGAII)search).search(wrapSE, trainSet,statNames);
		else{
			temp=new int[1][];
			if(search==null){
				temp[0]=new int[dataSet.numAttributes()-1];
				for(i=0;i<dataSet.numAttributes()-1;++i)
					temp[0][i]=i;
			}
			else
				temp[0]=search.search(wrapSE, trainSet);
		}
		
		for(i=0;i<temp.length;++i){
			int[] attrsIndex=new int[temp[i].length+1];
			for(j=0;j<temp[i].length;++j){
				attrsIndex[j]=temp[i][j];
				System.out.print(temp[i][j]);
				System.out.print(',');
			}
			System.out.println();
		
			attrsIndex[j]=dataSet.classIndex();
			Evaluation eval=new Evaluation(testSet);
			classifer.buildClassifier(trainSet.attributeFilter(attrsIndex));
			eval.evaluateModel(classifer, testSet.attributeFilter(attrsIndex));
			double[] result=new double[2];
			result[0]=attrsIndex.length-1;
			result[1]=eval.getPluginMetric("RegressionEval").getStatistic("MMRE");
			DenseInstance inst=new DenseInstance(1,result);
			results.add(inst);
		}
		return results;
	}
	
	private static void resultOut(Instances res,String searchName) throws IOException{
		fwResult=new File(outDir+searchName+".csv");
		CSVSaver csvSaver=new CSVSaver();
		csvSaver.setFile(fwResult);
		csvSaver.setInstances(res);
		csvSaver.writeBatch();
	}
	
	public static void main(String[] args)throws Exception{
		initialization();
		int i;
		Instances results;
		for(i=0;i<searchs.length;++i){
			System.out.println(searchNames[i]);
			System.out.println(results=experiment(searchs[i]));
			resultOut(results, searchNames[i]);
		}
	}
}

