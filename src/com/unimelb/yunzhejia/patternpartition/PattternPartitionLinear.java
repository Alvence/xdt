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
import com.yunzhejia.pattern.patternmining.RFPatternMiner;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

public class PattternPartitionLinear extends AbstractClassifier {

	Instances train;
	
	PatternSet patterns;
	
	Map<IPattern, double[]> A;
	
	
	double minSupp = 0.1;
	double minRatio = 3;
	int defaultClass=-1;
	
	double stepSize = 0.1;
	double beta = 0.5;
	@Override
	public void buildClassifier(Instances data) throws Exception {
		buildClassifierWithExpl(data,null);
	}
	
	public void buildClassifierWithExpl(Instances instances, Map<Long, Set<Integer>> expls) throws Exception {
		 instances = new Instances(instances);
		 instances.deleteWithMissingClass();
		 
		 IPatternMiner pm = new RFPatternMiner();
//			IPatternMiner pm = new GcGrowthPatternMiner(discretizer);
		 patterns = pm.minePattern(instances, minSupp);
		 
		 //initialize A
		 A = new HashMap<>();
		 for(IPattern p:patterns){
//			 Instances mds = p.matchingDataSet(instances);
//			 LinearRegression lr = new LinearRegression();
//			 lr.buildClassifier(mds);
//			 A.put(p, lr.coefficients());
			 A.put(p, new double[instances.numAttributes()]);
		 }
		 
		 //iteration
		 for(int t = 1; t < 200; t++){
			 System.out.println("t="+t);
			 showStat();
			 
			double[] temp = new double[instances.numInstances()];
			double[] coe = new double[instances.numAttributes()];
			for(int d = 0; d < coe.length; d++){
				coe[d] = 0;
				for(IPattern p : patterns){
					coe[d] += A.get(p)[d]; 
				}
			}
			 
			 for (int i = 0; i < instances.numInstances();i++){
				 Instance ins = instances.get(i);
				 double pred = 0;
				 for(int d = 0; d < coe.length; d++){
					 pred+= coe[d] * (d == ins.numAttributes()-1? 1:ins.value(d));
				 }
				 temp[i] = 2 * (ins.classValue() - pred);
			 }
			 
			 Map<IPattern, double[]> At = new HashMap<>();
			 
			 for(IPattern p : patterns){
				 double[] coet = new double[instances.numAttributes()];
				 for(int d = 0; d < coet.length; d++){
					 double der = 0;
					 for (int i = 0; i < instances.numInstances();i++){
						 Instance ins = instances.get(i);
						  der += temp[i] * (-1) *(p.match(ins)?1:0)*(d == ins.numAttributes()-1? 1:ins.value(d));
					 }
					 der = der/instances.numInstances();
					 coet[d] = A.get(p)[d] - stepSize*der;
				 }
				 At.put(p, coet);
			 }
			
			double[] coet2 = new double[instances.numAttributes()];
			for(int d = 0; d < coet2.length; d++){
				coet2[d] = 0;
				for(IPattern p : patterns){
					coet2[d] += At.get(p)[d]; 
				}
			}
			
			if(expls!=null){
				for (Instance ins: instances){
					if(!expls.containsKey(ins.getID())){
						continue;
					}
					Set<Integer> expl = expls.get(ins.getID());
					if(expl == null){
						continue;
					}
					for(IPattern p : patterns){
						for(int d = 0; d < coet2.length - 1; d++){
							if(expl.contains(d) && Math.abs(coet2[d])< 1e-7){
								At.get(p)[d] = At.get(p)[d] - beta* At.get(p)[d]/2; 
							}else if(!expl.contains(d) && Math.abs(coet2[d])>1e-7){
								At.get(p)[d] = At.get(p)[d] - beta* coet2[d] /2;
							}
						}
					}
				}
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
		
		return result;
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
			Instances train = DataUtils.load("data/modified/"+file+"_train.arff");
			Instances test = DataUtils.load("data/modified/"+file+"_test.arff");
			
			Map<Long, Set<Integer>> expls = ClassifierTruth.readFromFile("data/modified/expl/"+file+"_train.expl");
			
			PattternPartitionLinear cl = new PattternPartitionLinear();
			
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
