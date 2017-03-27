package SEE;

import weka.core.neighboursearch.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities.Capability;

import java.util.BitSet;

import org.netlib.util.intW;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.CapabilitiesIgnorer;
import weka.core.EuclideanDistance;

public class kNNRegression extends AbstractClassifier implements CapabilitiesIgnorer{
	private static final long serialVersionUID = 7307148053367392325L;
	
	private Instances m_train;
	private int m_k;
	private EuclideanDistance m_df;
	private LinearNNSearch m_LNN;
	
	@Override
	public boolean getDoNotCheckCapabilities(){
		return true;
	}
	
	public kNNRegression() throws Exception{
		this(null);
	}
	
	public kNNRegression(Instances insts) throws Exception{
		this(insts,5);
	}
	
	public kNNRegression(Instances insts,int k) throws Exception{
		m_df=new EuclideanDistance(insts);
		m_LNN=new LinearNNSearch();
		m_LNN.setDistanceFunction(m_df);
		setK(k);
	}

	@Override
	public void buildClassifier(Instances insts) throws Exception {
		m_train=insts;
		m_LNN.setInstances(m_train);
		if(m_k>insts.numInstances()){
			m_k=insts.numInstances();
			System.out.println("m_k is greater than numInstances.");
		}
	}

	@Override
	public double classifyInstance(Instance inst) throws Exception {
		Instances insts=new Instances(m_LNN.kNearestNeighbours(inst,m_k));
		return insts.meanOrMode(insts.classIndex());
	}

	@Override
	public double[] distributionForInstance(Instance inst) throws Exception {
		Instances insts=new Instances(m_LNN.kNearestNeighbours(inst,m_k));
		double pred[]={insts.meanOrMode(insts.classIndex())};
		return pred;
	}

	@Override
	public Capabilities getCapabilities() {
	    Capabilities result=new Capabilities(this);
	    // attributes
	    result.enable(Capability.NOMINAL_ATTRIBUTES);
	    result.enable(Capability.NUMERIC_ATTRIBUTES);
	    result.enable(Capability.DATE_ATTRIBUTES);
	    result.enable(Capability.STRING_ATTRIBUTES);
	    result.enable(Capability.MISSING_VALUES);

	    // class
	    result.enable(Capability.NOMINAL_CLASS);
	    result.enable(Capability.NUMERIC_CLASS);
	    result.enable(Capability.DATE_CLASS);
	    result.enable(Capability.STRING_CLASS);
	    result.enable(Capability.MISSING_CLASS_VALUES);
	    result.enable(Capability.NO_CLASS);
		return result;
	}
	
	public void setK(int k)throws Exception{
		if(k>0)
			m_k=k;
		else
			throw new Exception("kNNRegression's k must greater than 0.");
	}
	
	public void setAttributeIndices(int[] attrIndices){
		String range=String.valueOf(attrIndices[0]);
		for(int i=1;i<attrIndices.length;++i)
			range+=','+i;
		System.out.println(range);
		m_df.setAttributeIndices(range);
	}
	
	public void setAttributeIndices(String range){
		m_df.setAttributeIndices(range);
	}
}
