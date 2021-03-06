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

public class ExplPartitionWiseLinearModels_Random{

	public static void main(String[] args) throws Exception{
//		String[] files = { /*"adult",*/"balloon","blood"/*,"breast-cancer","chess"*/,"crx","diabetes","hepatitis",/*"ionosphere",*/
//				"labor","sick","vote"};
//		String[] files = {"anneal","balloon","blood","breast-cancer",/*"chess",*/"crx","diabetes","glass","hepatitis","ionosphere", "labor","sick","vote"};
//		String[] files = {"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
		String[] files = {"balloon","blood","diabetes","hepatitis", "labor", "vote"};
		
		ClassifierType[] types = {ClassifierType.DECISION_TREE};
		IPatternMiner[] pms = {new RFPatternMiner()};
		boolean[] flags = {true};
		int[] rands={20,40,60,80};
//		PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
		for(String file:files){
			for(ClassifierType type:types){
				for(IPatternMiner pm:pms){
					for(boolean flag:flags){
						for(int rand:rands){
			Instances train = DataUtils.load("data/modified"+rand+file+"_train.arff");
			Instances test = DataUtils.load("data/modified"+rand+file+"_test.arff");
//			Instances train = DataUtils.load("data/"+"synthetic_10samples.arff");
//			Instances test = DataUtils.load("data/"+"synthetic_10samples.arff");
			
			Map<Long, Set<Integer>> expls = ClassifierTruth.readFromFile("data/modified"+rand+"/expl/"+file+"_train.expl");
			Map<Long, Set<Integer>> explsTest = ClassifierTruth.readFromFile("data/modified"+rand+"/expl/"+file+"_test.expl");
			ExplPartitionWiseLinearModels cl = new ExplPartitionWiseLinearModels();
//			AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
			
			Evaluation eval = new Evaluation(test);
			
//			cl.buildClassifier(train);
//			cl.buildClassifierWithExpl(train, expls);
			cl.buildClassifierWithExpl(pm, train, flag?expls:null);
			eval.evaluateModel(cl, test);
			double losX = ExplEvaluation.evalExpl(cl,test,explsTest);
			
			System.out.println("data ="+ file +"rand="+rand+" flag="+flag+" accuracy="+ eval.pctCorrect()+"  losExpl="+losX);
			}}}}
		}
	}
}
