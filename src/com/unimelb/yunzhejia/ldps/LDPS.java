package com.unimelb.yunzhejia.ldps;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.cpxc.util.Discretizer;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.AprioriPatternMiner;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.RFPatternMiner;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class LDPS extends AbstractClassifier {

	Instances train;
	int K = 10;
	PatternSet ps;
	double minSupp = 0.10;
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		train = new Instances(data);
		
		Discretizer discretizer = new Discretizer();
		discretizer.initialize(train);
		IPatternMiner pm = new RFPatternMiner();
//		IPatternMiner pm = new AprioriPatternMiner(discretizer);
		
		ps = pm.minePattern(train, minSupp);
	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		
		Instances nei = findNearest(instance, K, train);
		
		AbstractClassifier cl = new J48();
		cl.buildClassifier(nei);
	
		
		return cl.distributionForInstance(instance);
	}

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
	
	public static void main(String[] args) throws Exception{
		String[] files = {/*"adult",*/"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
//		PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
		for(String file:files){
			Instances train = DataUtils.load("data/original/"+file+"_train.arff");
			Instances test = DataUtils.load("data/original/"+file+"_test.arff");
			
//			AbstractClassifier cl = new LDPS();
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierType.DECISION_TREE);
			cl.buildClassifier(train);
			
			Evaluation eval = new Evaluation(test);
			
			eval.evaluateModel(cl, test);
			
			System.out.println("data ="+ file +" accuracy="+ eval.pctCorrect());
		}
//		writer.close();
	}

}
