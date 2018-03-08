package com.unimelb.yunzhejia.patternpartition;

import java.util.Map;
import java.util.Set;

import com.unimelb.yunzhejia.xdt.ClassifierTruth;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.ParallelCoordinatesMiner;
import com.yunzhejia.pattern.patternmining.RFPatternMiner;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public class LR_benchmark {
	public static void main(String[] args) throws Exception{
//		String[] files = { /*"adult",*/"balloon","blood"/*,"breast-cancer","chess"*/,"crx","diabetes","hepatitis",/*"ionosphere",*/
//		"labor","sick","vote"};
//String[] files = {"anneal","balloon","blood","breast-cancer",/*"chess",*/"crx","diabetes","glass","hepatitis","ionosphere", "labor","sick","vote"};
//String[] files = {"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
String[] files = {"balloon","blood","diabetes","hepatitis", "labor", "vote","crx","sick"};

ClassifierType[] types = {ClassifierType.DECISION_TREE};
IPatternMiner[] pms = {new RFPatternMiner(), new ParallelCoordinatesMiner()};
boolean[] flags = { true, false};
//PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
for(String file:files){
	for(ClassifierType type:types){
		for(boolean flag:flags){
		for(IPatternMiner pm:pms){
	Instances train = DataUtils.load("data/modified/"+file+"_train.arff");
	Instances test = DataUtils.load("data/modified/"+file+"_test.arff");
//	Instances train = DataUtils.load("data/"+"synthetic_10samples.arff");
//	Instances test = DataUtils.load("data/"+"synthetic_10samples.arff");
	
	Map<Long, Set<Integer>> expls = ClassifierTruth.readFromFile("data/modified/expl/"+file+"_train.expl");
	Map<Long, Set<Integer>> explsTest = ClassifierTruth.readFromFile("data/modified/expl/"+file+"_test.expl");
	ExplPartitionWiseLinearModels cl = new ExplPartitionWiseLinearModels();
//	AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
	
	Evaluation eval = new Evaluation(test);
	
//	cl.buildClassifier(train);
//	cl.buildClassifierWithExpl(train, expls);
	cl.buildClassifierWithExpl(pm, train, flag?expls:null);
	eval.evaluateModel(cl, test);
	double losX = evalExpl(cl,test,explsTest);
	
	System.out.println("data ="+ file +"pm="+pm+" flag="+flag+" accuracy="+ eval.pctCorrect()+"  losExpl="+losX);
	}}}
}
	}

	private static double evalExpl(ExplPartitionWiseLinearModels cl, Instances test,
			Map<Long, Set<Integer>> explsTest) {
		// TODO Auto-generated method stub
		return 0;
	}
}
