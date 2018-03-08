package com.unimelb.yunzhejia.patternpartition;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import com.unimelb.yunzhejia.xdt.ClassifierTruth;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.MatchAllPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.ParallelCoordinatesMiner;
import com.yunzhejia.pattern.patternmining.RFPatternMiner;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class ExplPartitionWiseLinearModels extends AbstractClassifier {

	Instances train;
	
	PatternSet patterns;
	
	Map<IPattern, double[]> A;
	
	
	double minSupp = 0.2;
	double minRatio = 3;
	int defaultClass=-1;
	
	double stepSize = 0.1;
	double beta = 0.5;
	
	double the = 0.01;
	double gamma = 0.05;
	double lambdaP = 0.001;
	double lambda0 = 0.01; 
	/** The filter used to make attributes numeric. */
	public NominalToBinary m_NominalToBinary;

	  /** The filter used to get rid of missing values. */
	public ReplaceMissingValues m_ReplaceMissingValues;
	Random rand = new Random(0);
	
	int T = 3000;
	double C = 1;
	@Override
	public void buildClassifier(Instances data) throws Exception {
		buildClassifierWithExpl(new RFPatternMiner(),data,null);
	}
	
	public void buildClassifierWithExpl( IPatternMiner pm ,Instances train, Map<Long, Set<Integer>> expls) throws Exception {
		 Instances instances = new Instances(train);
		 instances.deleteWithMissingClass();
		 
		// Replace missing values
		 m_ReplaceMissingValues = new ReplaceMissingValues();
		 m_ReplaceMissingValues.setInputFormat(instances);
		 instances = Filter.useFilter(instances, m_ReplaceMissingValues);

		

		 // Transform attributes
		 m_NominalToBinary = new NominalToBinary();
		 m_NominalToBinary.setInputFormat(instances);
		 instances = Filter.useFilter(instances, m_NominalToBinary);
		 
		 
//		 IPatternMiner pm = new ManualPatternMiner();
//		 IPatternMiner pm = new RFPatternMiner();
//		 IPatternMiner pm = new ParallelCoordinatesMiner();
//		 IPatternMiner pm = new GcGrowthPatternMiner(discretizer);
		 patterns = pm.minePattern(instances, minSupp);

		 //make a global pattern that matches all instances
		 IPattern globalPattern = new MatchAllPattern();
		 patterns.add(globalPattern);
		 
//		 System.out.println(patterns);
//		 System.out.println(patterns.get(0).matchingDataSet(instances));
//		 System.out.println(patterns.get(1).matchingDataSet(instances));
//		 System.out.println(patterns.get(2).matchingDataSet(instances));
//		 if(true)
//		 return;
		 //initialize A
		 A = new HashMap<>();
		 for(IPattern p:patterns){
//			 Instances mds = p.matchingDataSet(instances);
//			 LinearRegression lr = new LinearRegression();
//			 lr.buildClassifier(mds);
//			 A.put(p, lr.coefficients());
			 double[] iniCoe = new double[instances.numAttributes()];
			 for(int i = 0; i < iniCoe.length; i++){
//				 iniCoe[i]=(int)(Math.random()*10+1);
				 iniCoe[i]=rand.nextDouble();
			 }
//			 System.out.println(p);
			 A.put(p, iniCoe);
		 }
	
		 double[] objs = new double[10];
		 int obj_index = 0;
		 
		 
		 //iteration
		 for(int t = 1; t < T ; t++){
//			 System.out.println("t="+t);
//			 showStat();
			 
			double[] preds = new double[instances.numInstances()];
			
			//calc current accuracy
			 
			int correct = 0;
			 for (int i = 0; i < instances.numInstances();i++){
				 Instance ins = instances.get(i);
				 //calculate errors
				 double[] coe = new double[instances.numAttributes()];
				 for(int d = 0; d < coe.length; d++){
						coe[d] = 0;
						for(IPattern p : patterns){
							if(p.match(ins)){
								coe[d] += A.get(p)[d];
							}
						}
				 }

				 double pred = 0;
				 for(int d = 0; d < coe.length; d++){
					 pred+= coe[d] * (d == ins.numAttributes()-1? 1:ins.value(d));
				 }
				 preds[i] = pred;
				 
				 int c = pred>0?1:0;
				 if(c == ins.classValue()){
					 correct++;
				 }
			 }
			 double acc = correct*1.0/instances.numInstances();
			 objs[obj_index%10] = acc;
			 obj_index++;
			 
			 //terminate condition
			 if(terminate(objs)){
				 break; 
			 }
			 
			 double[] explLoss = new double[instances.size()];
			 if(expls!=null){
			 for(int m = 0; m < instances.size(); m++){
				 Instance ins = instances.get(m);
				 if(expls.get((long)m) == null){
					 explLoss[m] = 0;
					 continue;
				 }
				 
				 double[] coe = new double[instances.numAttributes()];
				 double los = 0;
				 for(int dim = 0; dim < instances.numAttributes(); dim++){
					 coe[dim] = 0;
						for(IPattern p : patterns){
							if(p.match(ins)){
								coe[dim] += A.get(p)[dim];
							}
						}
					  double ed = (dim == ins.numAttributes()-1? 1 : (expls.get((long)m).contains(dim)?1:0)); 
							  
					  los += (coe[dim] - C*ed);
				 }
				 explLoss[m] = los;
			 }
			 }
			 
			 
//			 System.out.println("preds:  "+Arrays.toString(preds));
			 
			 Map<IPattern, double[]> At = new HashMap<>();
			 
			 
			 for(IPattern p : patterns){
				 double[] coet = new double[instances.numAttributes()];
				 
				 double[] coe_current = new double[instances.numAttributes()];
				 for(int k = 0; k < coe_current.length; k++){
					 coe_current[k] = A.get(p)[k];
				 }
				//M1 optimization
				 double theta = 0.0;
				 double total = 0.0;
				 int count = 0;
				 Arrays.sort(coe_current);
				 for(int i = coe_current.length-1; i>=0; i--){
					 double pivot = coe_current[i];
					 double sum = 0;
					 for(int j= coe_current.length-1; j > i; j--){
						 sum+= (Math.abs(coe_current[j])-Math.abs(coe_current[i]));
					 }
					 if(sum < lambdaP){
						 count++;;
						 total+= Math.abs(pivot);
					 }else{
						 break;
					 }
				 }
				 theta = (total-lambdaP)/count;
				 theta = (theta>0? theta:0);
				 
				 
				 
				 for(int d = 0; d < coet.length; d++){
					 double der = 0;
					 
					 double der2 = 0;
					 for (int i = 0; i < instances.numInstances();i++){
						 Instance ins = instances.get(i);
						 double y = ins.classValue();
						 der += (-1)*(y-(1/(1+Math.exp(-1*preds[i]))))  *(p.match(ins)?1:0)*(d == ins.numAttributes()-1? 1:ins.value(d));
					 }
					 
					 if(expls!=null){
					 for(int m = 0; m < instances.size();m++){
						 if (expls.get((long)m) == null){
							 continue;
						 }
						 Instance ins = instances.get(m);
						 der2+= explLoss[m] * (p.match(ins)?1:0);
					 }
					 }
//					 der = der/instances.numInstances();
//					 System.out.println(der);
					 coet[d] = A.get(p)[d] - stepSize*der;
					 
					 //loss expl
					 coet[d] = coet[d] - stepSize * gamma* der2;
//					 System.out.println(der);
					 
					 if(! (p instanceof MatchAllPattern)){
						 coet[d] = (coet[d]>0?1:-1)*Math.min(Math.abs(coet[d]),theta);
					 }
					 coet[d] = (coet[d]>0?1:-1)* (Math.abs(coet[d] - stepSize*lambda0)>0? Math.abs(coet[d] - stepSize*lambda0):0 );
					 
					 coet[d] = (Math.abs(coet[d])<1e-4?0:coet[d]);
				 }
				 At.put(p, coet);
			 }
			
			A = At;
		 }
//		 for(IPattern p:patterns){
//			 System.out.println(p+"   "+Arrays.toString( A.get(p)));
//		 }
	}
	
	private boolean terminate(double[] objs) {
		double dif = 0;
		for(int i = 0; i < objs.length-1; i++){
			dif = dif+Math.abs(objs[i+1]-objs[i]);
		}
		if(dif<1e-9){
			return true;
		}
		return false;
	}

	public void showStat(){
		for(IPattern p:patterns){
			System.out.println(p+":  " + Arrays.toString(A.get(p)));
		}
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		m_ReplaceMissingValues.input(instance);
	    instance = m_ReplaceMissingValues.output();
	    m_NominalToBinary.input(instance);
	    instance = m_NominalToBinary.output();
		
		double[] coe = new double[instance.numAttributes()];
		double result = 0;
		for(int d = 0; d < coe.length; d++){
			coe[d] = 0;
			for(IPattern p : patterns){
				if(p.match(instance)){
					coe[d] += A.get(p)[d];
				}
			}
			result+= coe[d] * (d == instance.numAttributes()-1? 1:instance.value(d));
		}
		
		return result>0?1:0;
	}
/*
	private Instances findNearest(Instance instance, int k, Instances headerInfo) {
		// TODO Auto-generated method stub
		Instances nei = new Instances(headerInfo,0);
		
		double[] distances = new double[train.size()];
		for(int i = 0; i < train.size(); i++){
			for(IPattern p:ps){
				if(p.match(instance)&&p.match(train.get(i))){
					distances[i] += p.support(train);
				}
			}
		}
		
		while(nei.size()<k){
			double large = -1;
			int index = -1;
			for(int i = 0;i < train.size();i++){
				if(distances[i]>large){
					large = distances[i];
					index = i;
				}
			}
			nei.add(train.get(index));
			distances[index] = -1;
		}
		
		
		return nei;
	}
	*/
	public static void main(String[] args) throws Exception{
//		String[] files = { /*"adult",*/"balloon","blood"/*,"breast-cancer","chess"*/,"crx","diabetes","hepatitis",/*"ionosphere",*/
//				"labor","sick","vote"};
//		String[] files = {"anneal","balloon","blood","breast-cancer",/*"chess",*/"crx","diabetes","glass","hepatitis","ionosphere", "labor","sick","vote"};
//		String[] files = {"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
		String[] files = {"balloon","blood","diabetes","hepatitis", "labor", "vote","crx","sick"};
		
		ClassifierType[] types = {ClassifierType.DECISION_TREE};
		IPatternMiner[] pms = {new RFPatternMiner(), new ParallelCoordinatesMiner()};
		boolean[] flags = { true, false};
//		PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
		for(String file:files){
			for(ClassifierType type:types){
				for(boolean flag:flags){
				for(IPatternMiner pm:pms){
			Instances train = DataUtils.load("data/modified/"+file+"_train.arff");
			Instances test = DataUtils.load("data/modified/"+file+"_test.arff");
//			Instances train = DataUtils.load("data/"+"synthetic_10samples.arff");
//			Instances test = DataUtils.load("data/"+"synthetic_10samples.arff");
			
			Map<Long, Set<Integer>> expls = ClassifierTruth.readFromFile("data/modified/expl/"+file+"_train.expl");
			Map<Long, Set<Integer>> explsTest = ClassifierTruth.readFromFile("data/modified/expl/"+file+"_test.expl");
			ExplPartitionWiseLinearModels cl = new ExplPartitionWiseLinearModels();
//			AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
			
			Evaluation eval = new Evaluation(test);
			
//			cl.buildClassifier(train);
//			cl.buildClassifierWithExpl(train, expls);
			cl.buildClassifierWithExpl(pm, train, flag?expls:null);
			eval.evaluateModel(cl, test);
			double losX = ExplEvaluation.evalExpl(cl,test,explsTest);
			
			System.out.println("data ="+ file +"pm="+pm+" flag="+flag+" accuracy="+ eval.pctCorrect()+"  losExpl="+losX);
			}}}
		}
	}
	
	public static double evalExpl(ExplPartitionWiseLinearModels cl, Instances data, Map<Long, Set<Integer>> expls) throws Exception{
		double ret = 0;
		
		for(int i = 0; i < data.numInstances();i++){
			Instance ins = data.get(i);
			
			cl.m_ReplaceMissingValues.input(ins);
			ins = cl.m_ReplaceMissingValues.output();
		    cl.m_NominalToBinary.input(ins);
		    ins = cl.m_NominalToBinary.output();
			
			
			
			 double[] coe = new double[data.numAttributes()];
			 for(int dim = 0; dim < data.numAttributes(); dim++){
				 coe[dim] = 0;
					for(IPattern p : cl.patterns){
						if(p.match(ins)){
							coe[dim] += cl.A.get(p)[dim];
						}
					}
			 }
			 Utils.normalize(coe);
			 for(int dim = 0; dim < data.numAttributes(); dim++){
				  double ed = (dim == ins.numAttributes()-1? 1 : (expls.get((long)i).contains(dim)?1:0)); 
				 
				  ret += (coe[dim] - cl.C*ed)*(coe[dim] - cl.C*ed);
//				  ret += (Math.abs(coe[dim])>cl.the?1:0 - ed)*(Math.abs(coe[dim])>cl.the?1:0 - ed);
			 }
		}
		
		return ret/data.numInstances();
	}

}
