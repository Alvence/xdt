package com.unimelb.yunzhejia.patternpartition;

import java.util.List;
import java.util.Map;
import java.util.Set;

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
import weka.core.Utils;

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
		 for(IPattern p:patterns){
			 Instances mds = p.matchingDataSet(instances);
			 LinearRegression lr = new LinearRegression();
			 lr.buildClassifier(mds);
			 A.put(p, lr.coefficients());
		 }
		 
		 //iteration
		 for(int t = 1; t < 20; t++){
			 double temp = 0;
			 double[] coe = new double[instances.numAttributes()];
			 for(int d = 0; d < coe.length; d++){
					coe[d] = 0;
					for(IPattern p : patterns){
						coe[d] += A.get(p)[d]; 
					}
				}
			 
			 for (Instance ins: instances){
				 double pred = 0;
				 for(int d = 0; d < coe.length; d++){
					 pred+= coe[d] * (d == ins.numAttributes()-1? 1:ins.value(d));
				 }
				 temp += 2 * (ins.classValue() - pred);
			 }
			 
			 
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
		String[] files = {/*"adult","anneal",*/"balloon","blood","breast-cancer","diabetes","ILPD","iris","labor","vote","hepatitis","ionosphere"};
//		ClassifierType[] types = {ClassifierType.DECISION_TREE, ClassifierType.LOGISTIC, ClassifierType.NAIVE_BAYES, ClassifierType.RANDOM_FOREST};
		ClassifierType[] types = {ClassifierType.DECISION_TREE};
//		PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
		for(String file:files){
			for(ClassifierType type:types){
			Instances train = DataUtils.load("data/original/"+file+"_train.arff");
			Instances test = DataUtils.load("data/original/"+file+"_test.arff");
			train.deleteAttributeAt(2);
//			AbstractClassifier cl = new LDPS();
			AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
			cl.buildClassifier(train);
			
			Evaluation eval = new Evaluation(test);
			test.deleteAttributeAt(2);
			eval.evaluateModel(cl, test);
			
			System.out.println("data ="+ file +" accuracy="+ eval.pctCorrect());
		}}
//		writer.close();
	}

}
