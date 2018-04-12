package com.unimelb.yunzhejia.patternpartition;

import java.util.Map;
import java.util.Set;

import com.unimelb.yunzhejia.xdt.ClassifierTruth;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class LR_benchmark {
	public static void main(String[] args) throws Exception{
//		String[] files = { /*"adult",*/"balloon","blood"/*,"breast-cancer","chess"*/,"crx","diabetes","hepatitis",/*"ionosphere",*/
//		"labor","sick","vote"};
//String[] files = {"anneal","balloon","blood","breast-cancer",/*"chess",*/"crx","diabetes","glass","hepatitis","ionosphere", "labor","sick","vote"};
//String[] files = {"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
String[] files = {"balloon","blood","crx","diabetes","hepatitis","hypo", "labor","sick","titanic","vote"};

//PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
for(String file:files){
	Instances train = DataUtils.load("data/noisy50/"+file+"_train.arff");
	Instances test = DataUtils.load("data/noisy50/"+file+"_test.arff");
//	Instances train = DataUtils.load("data/"+"synthetic_10samples.arff");
//	Instances test = DataUtils.load("data/"+"synthetic_10samples.arff");
	
	Map<Long, Set<Integer>> expls = ClassifierTruth.readFromFile("data/noisy50/expl/"+file+"_train.expl");
	Map<Long, Set<Integer>> explsTest = ClassifierTruth.readFromFile("data/noisy50/expl/"+file+"_test.expl");
//	ExplPartitionWiseLinearModels cl = new ExplPartitionWiseLinearModels();
	AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierGenerator.ClassifierType.NN);
//	Logistic cl = new Logistic();
	
	Evaluation eval = new Evaluation(test);
	
	cl.buildClassifier(train);
//	cl.buildClassifierWithExpl(train, expls);
//	cl.buildClassifierWithExpl(pm, train, flag?expls:null);
	eval.evaluateModel(cl, test);
	double losX = ExplEvaluation.evalExpl(cl,test,explsTest);
	
	System.out.println("data ="+ file +" accuracy="+ eval.pctCorrect()+"  losExpl="+losX);

}
	}
 }
