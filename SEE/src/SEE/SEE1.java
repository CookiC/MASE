package SEE;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
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
import weka.core.Utils;
import weka.core.WekaPackageManager;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class SEE1 {
	private static Instances dataSet;
	
	private static Instances trainSet;
	
	private static Instances testSet;
	
	private static final String[] searchNames={"FSFS","BSFS","SOFS","NOFS","MASE"};
	
	private static final String outDir="E:/MachineLearning/ExperimentData/maxwell/";
	
	private static final String[] dataSetNames={"maxwell.arff"};
	
	private static final String[] statNames={"FN","MMRE"};
	//Data in.
	private static File frData;
	//Result out.
	private static File fwResult;
	
	private static ASSearch[] searchs=new ASSearch[5];
	
	private static ArrayList<Attribute> resAttrs=new ArrayList<Attribute>();
	
	private static FileWriter fw;
	
	private static void initialization() throws Exception{
		fw=new FileWriter(outDir+"box.csv");
		PluginManager.addPlugin(AbstractEvaluationMetric.class.getName(),"RegressionEval",RegressionEval.class.getName());
		frData=new File(dataSetNames[0]);
		dataSet = new Instances(new FileReader(frData));
		dataSet.setClassIndex("Effort");
		dataSet.sort(0);
		dataSet.deleteAttributeAt(0);
		trainSet = new Instances(dataSet, 0, 50);
        testSet = new Instances(dataSet, 50, 12);
		//dataSet.deleteAttributeAt(17);
		//dataSet.deleteAttributeAt(0);
		//randomSplit(dataSet, 0.7);
		GreedyStepwise FW=new GreedyStepwise();
		FW.setSearchBackwards(false);
		GreedyStepwise BW=new GreedyStepwise();
		BW.setSearchBackwards(true);
		GeneticSearch GS=new GeneticSearch();
		GS.setMaxGenerations(100);
		GS.setMutationProb(0.1);
		GS.setPopulationSize(100);
		NSGAII nsgaii=new NSGAII();
		nsgaii.setMaxGenerations(100);
		nsgaii.setMutationProb(0.1);
		nsgaii.setPopulationSize(100);
		searchs[0]=FW;
		searchs[1]=BW;
		searchs[2]=GS;
		searchs[3]=null;
		searchs[4]=nsgaii;
		resAttrs.add(new Attribute("FN"));
		resAttrs.add(new Attribute("MMRE"));
	}
	
	private static void randomSplit(Instances dataSet, double d) throws Exception {
		dataSet.randomize(new Random(System.currentTimeMillis()));
		int dataSize = dataSet.numInstances();
		int trainSize = (int)Math.round(dataSize * d);
        int testSize = dataSize - trainSize;
        trainSet = new Instances(dataSet, 0, trainSize);
        testSet = new Instances(dataSet, trainSize, testSize);
	}
	
	private static double[] experiment(ASSearch search,Instances results) throws Exception {
		int i,j;
		kNNRegression classifer=new kNNRegression();
		SingleWrapperSubsetEval wrapSE=new SingleWrapperSubsetEval();
		SelectedTag evalMeasure=new SelectedTag("MMRE",WrapperSubsetEval.TAGS_EVALUATION);
		classifer.setK(3);
		wrapSE.setClassifier(classifer);
		wrapSE.setEvaluationMeasure(evalMeasure);
		wrapSE.setFolds(3);
		wrapSE.buildEvaluator(trainSet);
		int[][] temp;
		if (search instanceof NSGAII)
			temp=((NSGAII)search).search(wrapSE, trainSet,statNames);
		else{
			temp=new int[1][];
			if(search==null){
				temp[0]=new int[dataSet.numAttributes()-1];
				for(i=0,j=0;i<dataSet.numAttributes();++i)
					if(i!=dataSet.classIndex())
						temp[0][j++]=i;
			}
			else
				temp[0]=search.search(wrapSE, trainSet);
		}
		
		double minMMRE=100;
		double[] MREs=null;
		for(i=0;i<temp.length;++i){
			int[] attrsIndex=new int[temp[i].length+1];
			for(j=0;j<temp[i].length;++j){
				attrsIndex[j]=temp[i][j];
				System.out.print(temp[i][j]);
				System.out.print(',');
			}
			System.out.println();
		
			attrsIndex[j]=dataSet.classIndex();
			Instances trainCopy=trainSet.attributeFilter(attrsIndex);
			Instances testCopy=testSet.attributeFilter(attrsIndex);
			Evaluation eval=new Evaluation(trainCopy);
			classifer.buildClassifier(trainCopy);
			eval.evaluateModel(classifer, trainCopy);
			double[] result=new double[2];
			AbstractEvaluationMetric metric=eval.getPluginMetric("RegressionEval");
			result[0]=attrsIndex.length-1;
			result[1]=metric.getStatistic("MMRE");
			
			eval=new Evaluation(trainCopy);
			eval.evaluateModel(classifer, testCopy);
			metric=eval.getPluginMetric("RegressionEval");
			System.out.println("FN:"+(metric.getStatistic("FN")-1)+",MMRE:"+metric.getStatistic("MMRE")+",MdMRE:"+metric.getStatistic("MdMRE")+",pred(0.25):"+metric.getStatistic("predR"));
			if(metric instanceof RegressionEval&&minMMRE>metric.getStatistic("MMRE")){
				MREs=((RegressionEval) metric).getMREs();
				minMMRE=metric.getStatistic("MMRE");
			}
			DenseInstance inst=new DenseInstance(1,result);
			results.add(inst);
		}
		return MREs;
	}
	
	private static void resultOut(Instances res,String searchName) throws IOException{
		fwResult=new File(outDir+searchName+".csv");
		CSVSaver csvSaver=new CSVSaver();
		csvSaver.setFile(fwResult);
		csvSaver.setInstances(res);
		csvSaver.writeBatch();
	}
	
	private static void resultOut(double[] MREs,String name) throws IOException {
		String out="";
		for(int i=0;i<MREs.length;++i)
			out+=MREs[i]+","+name+"\n";
		fw.write(out);
	}
	
	public static void main(String[] args)throws Exception{
		initialization();
		int i,j;
		long start;
		Instances results;
		for(i=0;i<searchs.length;++i){
			results=new Instances("results",resAttrs,10);
			start=System.currentTimeMillis();
			System.out.println(searchNames[i]);
			resultOut(experiment(searchs[i],results),searchNames[i]);
			System.out.println((System.currentTimeMillis()-start)+"ms");
			resultOut(results, searchNames[i]);
		}
		fw.close();
	}
}

