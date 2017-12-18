package com.unimelb.yunzhejia.patternpartition;

import java.util.Map;
import java.util.Set;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.PatternSet;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class PattternPartitionLinear extends AbstractClassifier {

	Instances train;
	int K = 10;
	Map<Integer, PatternSet> patterns;
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
		
		
	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		
		double[] probs = new double[train.numClasses()];
		
		Utils.normalize(probs);
		
		return probs;
		/*
		Instances nei = findNearest(instance, K, train);
		
		AbstractClassifier cl = new J48();
		cl.buildClassifier(nei);
	
		
		return cl.distributionForInstance(instance);*/
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
