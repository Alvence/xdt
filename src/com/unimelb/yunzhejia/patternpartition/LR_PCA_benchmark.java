package com.unimelb.yunzhejia.patternpartition;

import java.util.Map;
import java.util.Set;

import com.unimelb.yunzhejia.xdt.ClassifierTruth;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.ParallelCoordinatesMiner;
import com.yunzhejia.pattern.patternmining.RFPatternMiner;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Debug.Log;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.core.Instances;

public class LR_PCA_benchmark {
	public static void main(String[] args) throws Exception{
//		String[] files = { /*"adult",*/"balloon","blood"/*,"breast-cancer","chess"*/,"crx","diabetes","hepatitis",/*"ionosphere",*/
//		"labor","sick","vote"};
//String[] files = {"anneal","balloon","blood","breast-cancer",/*"chess",*/"crx","diabetes","glass","hepatitis","ionosphere", "labor","sick","vote"};
//String[] files = {"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
String[] files = {"balloon","blood","crx","diabetes","hepatitis", "labor","sick", "vote"};

//PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
for(String file:files){
	Instances train = DataUtils.load("data/noisy50/"+file+"_train.arff");
	Instances test = DataUtils.load("data/noisy50/"+file+"_test.arff");
//	Instances train = DataUtils.load("data/"+"synthetic_10samples.arff");
//	Instances test = DataUtils.load("data/"+"synthetic_10samples.arff");
	
	PrincipalComponents pca = new PrincipalComponents();
	pca.setInputFormat(train);
	pca.m_MaxAttributes = 5;
	Instances newTrain = Filter.useFilter(train, pca);
	Instances newTest = Filter.useFilter(test, pca);
	
	
	Map<Long, Set<Integer>> expls = ClassifierTruth.readFromFile("data/noisy50/expl/"+file+"_train.expl");
	Map<Long, Set<Integer>> explsTest = ClassifierTruth.readFromFile("data/noisy50/expl/"+file+"_test.expl");
//	ExplPartitionWiseLinearModels cl = new ExplPartitionWiseLinearModels();
//	AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
	Logistic cl = new Logistic();
	
	Evaluation eval = new Evaluation(newTest);
	
	cl.buildClassifier(newTrain);
//	cl.buildClassifierWithExpl(train, expls);
//	cl.buildClassifierWithExpl(pm, train, flag?expls:null);
	eval.evaluateModel(cl, newTest);
	double losX = ExplEvaluation.evalExpl(cl,pca,newTest,explsTest);
	
	System.out.println("data ="+ file +" accuracy="+ eval.pctCorrect()+"  losExpl="+losX);

}
	}
 }
