package com.unimelb.yunzhejia.patternpartition;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.unimelb.yunzhejia.xdt.ClassifierTruth;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.ManualPatternMiner;
import com.yunzhejia.pattern.patternmining.RFPatternMiner;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

public class PartitionWiseLinearModels extends AbstractClassifier {

	Instances train;
	
	PatternSet patterns;
	
	Map<IPattern, double[]> A;
	
	
	double minSupp = 0.1;
	double minRatio = 3;
	int defaultClass=-1;
	
	double stepSize = 0.1;
	double beta = 0.5;
	
	double lambdaP = 0.001;
	double lambda0 = 0.001; 
	
	int T = 30;
	@Override
	public void buildClassifier(Instances data) throws Exception {
		buildClassifierWithExpl(data,null);
	}
	
	public void buildClassifierWithExpl(Instances instances, Map<Long, Set<Integer>> expls) throws Exception {
		 instances = new Instances(instances);
		 instances.deleteWithMissingClass();
		 IPatternMiner pm = new ManualPatternMiner();
//		 IPatternMiner pm = new RFPatternMiner();
//			IPatternMiner pm = new GcGrowthPatternMiner(discretizer);
		 patterns = pm.minePattern(instances, minSupp);
		 
//		 System.out.println(patterns);
//		 System.out.println(patterns.get(0).matchingDataSet(instances));
//		 System.out.println(patterns.get(1).matchingDataSet(instances));
//		 System.out.println(patterns.get(2).matchingDataSet(instances));
//		 if(true)
//		 return;
		 //initialize A
		 A = new HashMap<>();
		 for(IPattern p:patterns){
			 Instances mds = p.matchingDataSet(instances);
//			 LinearRegression lr = new LinearRegression();
//			 lr.buildClassifier(mds);
//			 A.put(p, lr.coefficients());
			 double[] iniCoe = new double[instances.numAttributes()];
			 for(int i = 0; i < iniCoe.length; i++){
				 iniCoe[i]=(int)(Math.random()*10+1);
			 }
			 A.put(p, iniCoe);
		 }
		 
		 //iteration
		 for(int t = 1; t < T ; t++){
			 System.out.println("t="+t);
			 showStat();
			 
			double[] errs = new double[instances.numInstances()];
			
			 
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
				 errs[i] = (ins.classValue() - pred);
			 }
			 
			 System.out.println("errors:  "+Arrays.toString(errs));
			 
			 Map<IPattern, double[]> At = new HashMap<>();
			 
			 for(IPattern p : patterns){
				 double[] coet = new double[instances.numAttributes()];
				 for(int d = 0; d < coet.length; d++){
					 double der = 0;
					 for (int i = 0; i < instances.numInstances();i++){
						 Instance ins = instances.get(i);
						  der += errs[i]  *(p.match(ins)?1:0)*(d == ins.numAttributes()-1? 1:ins.value(d));
					 }
//					 der = der/instances.numInstances();
					 coet[d] = A.get(p)[d] - stepSize*der;
					 
					 
//					 coet[d] = (coet[d]>0?1:-1)*(Math.abs(coet[d]) - stepSize*lambdaP);
					 
//					 coet[d] = (coet[d]>0?1:-1)* (Math.abs(coet[d] - stepSize*lambda0)>0? Math.abs(coet[d] - stepSize*lambda0):0 );
				 }
				 At.put(p, coet);
			 }
			
			
			A = At;
		 }
	}
	
	public void showStat(){
		for(IPattern p:patterns){
			System.out.println(p+":  " + Arrays.toString(A.get(p)));
		}
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double[] coe = new double[instance.numAttributes()];
		double result = 0;
		for(int d = 0; d < coe.length; d++){
			coe[d] = 0;
			for(IPattern p : patterns){
				coe[d] += A.get(p)[d]; 
			}
			result+= coe[d] * (d == instance.numAttributes()-1? 1:instance.value(d));
		}
		
		return result>0.5?1:0;
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
//		String[] files = {"anneal","balloon","blood","breast-cancer","chess","crx","diabetes","glass","hepatitis","ionosphere"
//				,"iris","labor","sick","vote"};
//		String[] files = {"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
		String[] files = {"iris"};
		
		ClassifierType[] types = {ClassifierType.DECISION_TREE};
//		PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
		for(String file:files){
			for(ClassifierType type:types){
//			Instances train = DataUtils.load("data/modified/"+file+"_train.arff");
//			Instances test = DataUtils.load("data/modified/"+file+"_test.arff");
			Instances train = DataUtils.load("data/"+"synthetic_10samples.arff");
			Instances test = DataUtils.load("data/"+"synthetic_10samples.arff");
			
			Map<Long, Set<Integer>> expls = ClassifierTruth.readFromFile("data/modified/expl/"+file+"_train.expl");
			
			PartitionWiseLinearModels cl = new PartitionWiseLinearModels();
			
			Evaluation eval = new Evaluation(test);
			
//			cl.buildClassifier(train);
//			cl.buildClassifierWithExpl(train, expls);
			cl.buildClassifierWithExpl(train, null);
			eval.evaluateModel(cl, test);
			
			System.out.println("data ="+ file +" accuracy="+ eval.pctCorrect());
			}
		}
	}

}
