package com.unimelb.yunzhejia.ldps;

import java.util.HashMap;
import java.util.Map;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.cpxc.util.Discretizer;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.AprioriContrastPatternMiner;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.RFContrastPatternMiner;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class LDPS_KNN extends AbstractClassifier {

	Instances train;
	int K = 10;
	Map<Integer, PatternSet> patterns;
	double minSupp = 0.01;
	double minRatio = 5;
	int defaultClass=-1;
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		train = new Instances(data);
		
		Discretizer discretizer = new Discretizer();
		discretizer.initialize(train);
//		IPatternMiner pm = new RFPatternMiner();
//		IPatternMiner pm = new AprioriPatternMiner(discretizer);
//		IPatternMiner pm = new AprioriContrastPatternMiner(discretizer);
		IPatternMiner pm = new RFContrastPatternMiner();
		int maxSize = -1;
		
		patterns = new HashMap<>();
		for(int i = 0; i < data.numClasses(); i++){
			PatternSet ps = pm.minePattern(train, minSupp, minRatio, i);
			System.out.println(i+ ": "+ps.size());
			patterns.put(i, ps);
			if(maxSize<ps.size()){
				maxSize = ps.size();
				defaultClass = i;
			}
		}
		
	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		
		double[] probs = new double[train.numClasses()];
		boolean found = false;
		for(int i =0; i < train.numClasses(); i++){
			PatternSet ps = patterns.get(i);
			for(IPattern p: ps){
				if (p.match(instance)){
					probs[i] += 1;
					found = true;
				}
			}
		}
		
		if(!found){
			probs[defaultClass] = 1;
		}
		
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
//		String[] files = {/*"adult","anneal",*/"balloon","blood","breast-cancer","diabetes","ILPD","iris","labor","vote","hepatitis","ionosphere"};
		String[] files = {/*"adult","anneal",*/"balloon"};
//		ClassifierType[] types = {ClassifierType.DECISION_TREE, ClassifierType.LOGISTIC, ClassifierType.NAIVE_BAYES, ClassifierType.RANDOM_FOREST};
		ClassifierType[] types = {ClassifierType.DECISION_TREE};
//		PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
		for(String file:files){
			for(ClassifierType type:types){
			Instances train = DataUtils.load("data/original/"+file+"_train.arff");
			Instances test = DataUtils.load("data/original/"+file+"_test.arff");
			
			AbstractClassifier cl = new MY_LWL();
//			AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
			cl.buildClassifier(train);
			
//			Evaluation eval = new Evaluation(test);
//			
//			eval.evaluateModel(cl, test);
//			
//			System.out.println("data ="+ file +" accuracy="+ eval.pctCorrect());
		}}
//		writer.close();
	}

}
